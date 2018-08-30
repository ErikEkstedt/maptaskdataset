import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchaudio

from torchaudio.transforms import F2M, SPEC2DB, MEL2
from torchaudio.transforms import SPECTROGRAM, Compose, _check_is_variable

from maptask.maptaskdata import MaptaskDataset
from maptask.utils import sound_torch_tensor

from librosa.filters import mel as librosa_mel_filter
from librosa.filters import mel as librosa_mel_fn
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import librosa.util as librosa_util
from librosa.util import pad_center, tiny
from scipy.signal import get_window


# ------------------
def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(win_length >= filter_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

# ------------------------

def show_spec(s, y_axis='mel', title=''):
    plt.figure()
    librosa.display.specshow(s.numpy().T, sr=20000, y_axis=y_axis)
    plt.title(title)
    plt.pause(0.1)


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


class Filter(object):
    def __init__(self, n_mels=40, sr=20000, f_max=None, f_min=0.):
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min

    def F2M(self, spec_f):
        spec_f, is_variable = _check_is_variable(spec_f)
        n_fft = spec_f.size(2)

        m_min = 0. if self.f_min == 0 else 2595 * np.log10(1. + (self.f_min / 700))
        m_max = 2595 * np.log10(1. + (self.f_max / 700))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = (700 * (10**(m_pts / 2595) - 1))

        bins = torch.floor(((n_fft - 1) * 2) * f_pts / self.sr).long()

        fb = torch.zeros(n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)

        self.fb = fb
        spec_m = torch.matmul(spec_f, fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m if is_variable else spec_m.data

    def M2F(self, mels):
        fb_t = self.fb.t()
        fb_sum = self.fb.sum(dim=1) + 1e-5
        print('fb_sum shape: ', fb_sum.shape)
        print('fb_sum:', fb_sum)

        print('fb_t shape: ', fb_t.shape)
        print('fb_t:', fb_t)

        fb_inv = fb_t / fb_sum
        print('fb_inv shape: ', fb_inv.shape)
        print('fb_inv', fb_inv)
        # spec_m = torch.matmul(fb_inv, mels)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        spec_m = torch.matmul(mels, fb_inv)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m 


if __name__ == "__main__":

    # Load test audio to torch.tensor
    dset = MaptaskDataset(pause=0.5, max_utterences=1, context=2)

    # one short audio
    y = dset[80]['context_audio']
    sound_torch_tensor(y)
    y = y.unsqueeze(0)

    # STFTS
    mel = MEL2(sr=20000, ws=512, hop=256, pad=0, window=torch.hann_window)
    m = mel(y)
    show_spec(m.squeeze(0), y_axis='mel', title='spec')

    filt = Filter(n_mels=40, sr=20000)

    spec = SPECTROGRAM(sr=20000, ws=512, hop=256, pad=0, window=torch.hann_window)
    s_db = SPEC2DB(stype='power', top_db=-80.)
    
    s = spec(y)
    m = filt.F2M(s)

    mels = s_db(m)

    si = filt.M2F(m)
    si = s_db(si)

    show_spec(s.squeeze(0), y_axis='linear', title='spec')
    show_spec(mels.squeeze(0), y_axis='mel', title='MEL')

    # Filters
    stft_fn = STFT(filter_length=512, hop_length=256, win_length=512)

    mel_matrix = librosa_mel_filter(sr=20000, n_fft=512, n_mels=40, fmin=0.0, fmax=None, htk=True, norm=1)
    mel_mat = torch.from_numpy(mel_matrix).float()


    spec_f = torch.stft(y,
                        n_fft=512,
                        hop_length=256,
                        win_length=512,
                        center=False,
                        normalized=False,
                        onesided=True).transpose(1, 2)
    s = spec_f[:,:,:,0]

    mel_filter = MelFilter(n_fft=257, n_mels=40, sr=20000, f_max=None, f_min=0.)
    mels = mel_filter(s)
    mels = s_db(mels).squeeze(0)

    librosa.display.specshow(mels.numpy(), sr=20000, y_axis='mel')


    ms = torch.matmul(s, mel_mat.t())
    ms = s_db(ms)

    show_spec(s.squeeze(0), y_axis='mel', title='Melspec')
    show_spec(s.squeeze(0), y_axis='log', title='Spec')

    # Spectrogram
    mels = mel(y).squeeze(0)  # 1,155,40 -> 155,40
    s = spec(y).squeeze(0)  # 1,155,257 -> 155,257
    s, phase = spec_f[:,:,:,0], spec_f[:,:,:,1]
    s = s.permute(0,2,1)

    y_recon = griffin_lim(s, stft_fn)
    sound_torch_tensor(y_recon.squeeze(0))


    # tacotron
    s2, phase = stft_fn.transform(y)
    y_recon = griffin_lim(s2, stft_fn)
    sound_torch_tensor(y_recon.squeeze(0))

    ms = torch.matmul(s, mel_mat.t())
    ms2 = torch.matmul(mel_mat, s.t())

    # Plot
    show_spec(mels.squeeze(0), y_axis='mel', title='Melspec')
    show_spec(s.squeeze(0), y_axis='log', title='Spec')

    show_spec(s.squeeze(0), y_axis='linear', title='Spec Linear')

    show_spec(s2.squeeze(0), y_axis='linear', title='Spec Linear')
