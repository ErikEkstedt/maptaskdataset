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


# -----------------------------
def hz2mel(f):
    return 1125 * np.log(1. + (f / 700))


def mel2hz(m):
    return 700 *(np.exp(m / 1125) - 1)


class Filter(object):
    def __init__(self, n_mels=40, sr=20000, f_max=None, f_min=0.):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min

        self.m_min = self.hz2mel(f_min)
        self.m_max = self.hz2mel(f_max)
        self.m_pts = self._mel_filter_pts()
        self.f_pts = self.mel2hz(self.m_pts)

        self.filter_bank = self._filter_bank()

    def _mel_filter_pts(self):
        return torch.linspace(self.m_min, self.m_max, self.n_mels + 2)

    def _filter_bank(self, n_fft):
        bins = torch.floor((n_fft + 1) * self.f_pts / self.sr).long()

        fb = torch.zeros(n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)

    def F2M(self, spec_f):
        n_fft = spec_f.size(2)


        self.fb = fb
        spec_m = torch.matmul(spec_f, fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m if is_variable else spec_m.data

    def M2F(self, mels):
        return mels 


class Spectrogram(object):
    """Create a spectrogram from a raw audio signal
    Args:
        sr (int): sample rate of audio signal
        ws (int): window size, often called the fft size as well
        hop (int, optional): length of hop between STFT windows. default: ws // 2
        n_fft (int, optional): number of fft bins. default: ws // 2 + 1
        pad (int): two sided padding of signal
        window (torch windowing function): default: torch.hann_window
        wkwargs (dict, optional): arguments for window function

    """
    def __init__(self, sr=20000, ws=500, hop=None, n_fft=None,
                 pad=0, window=torch.hann_window, wkwargs=None):
        self.sr = sr
        self.ws = ws
        self.hop = hop if hop is not None else ws // 2
        # number of fft bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.n_fft = (n_fft - 1) * 2 if n_fft is not None else ws
        self.pad = pad
        self.wkwargs = wkwargs

        if isinstance(window, torch.Tensor):
            self.window = window
        else:
            self.window = window(ws) if wkwargs is None else window(ws, **wkwargs)

    def pad(self, sig):
        c, n = sig.size()
        new_sig = sig.new_empty(c, n + self.pad * 2)
        new_sig[:, :self.pad].zero_()
        new_sig[:, -self.pad:].zero_()
        new_sig.narrow(1, self.pad, n).copy_(sig)
        return new_sig

    def __call__(self, sig):
        assert sig.dim() == 2
        if self.pad > 0:
            sig = self.pad(sig)

        s = torch.stft(sig, self.n_fft, self.hop,
                       self.ws, self.window, center=False,
                       normalized=True, onesided=True).transpose(1, 2)
        return s 


def play_array(x, sr=20000):
    assert x.max() <= 1 
    sd.default.samplerate=sr
    sd.play(x)


# min/max freqs
f_min = 300
f_max = 8000
n_mels = 10
n_fft = 500
sr = 20000
ws = 500  # sr*25ms
hop = 200  # sr*10ms

# Get mel-frequency max, min
m_min = hz2mel(f_min)
m_max = hz2mel(f_max)

# Get
m_pts = torch.linspace(m_min, m_max, n_mels + 2)
f_pts = mel2hz(m_pts)

bins = torch.floor( (n_fft + 1) * f_pts / sr).long()

fb = torch.zeros(n_fft, n_mels)
for m in range(1, n_mels + 1):
    f_m_minus = bins[m - 1].item()  # left
    f_m = bins[m].item()  # center
    f_m_plus = bins[m + 1].item()  # right

    if f_m_minus != f_m:
        fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
    if f_m != f_m_plus:
        fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
spec_m = torch.matmul(spec_f, fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)


from librosa.filters import mel
from librosa.core.spectrum import _spectrogram, power_to_db


# librosa mel
y_, sr = torchaudio.load('/Users/erik/maptaskdataset/maptask/data/dialogues/q1ec1.mix.wav')
y = y_[:40000, 0].unsqueeze(0)
y /= y.max()
y = y.contiguous()
yn = y.squeeze(0).numpy()

melfilter = mel(sr, n_fft, n_mels, f_min, f_max)

spec = Spectrogram(sr=sr, ws=ws, hop=hop, n_fft=None, pad=0, window=torch.hann_window)

S = np.abs(librosa.core.stft(yn, n_fft, hop, center=False,window='hann'))**2


sn = librosa.core.stft(yn, n_fft, hop, center=False,window='hann')

s = torch.stft(y, n_fft, hop, center=False)
s = s.squeeze(0)
s = torch.sqrt( s[:,:,0]**2 + s[:,:,1]**2)

s = s.pow(2)

np.dot(s, melfilter)

s = np.abs(spec(y))**2

s = s.squeeze(0)

s = torch.sqrt( s[:,:,0].pow(2) + s[:,:,1].pow(2))


librosa.display.specshow(s.numpy().T)
plt.show()


