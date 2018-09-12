# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# import numpy as np
# import os
# import audio

from nnmnkwii.datasets import FileDataSource
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists, join, basename, splitext, exists
from glob import glob
from tqdm import tqdm
import numpy as np

import pysptk
import pyworld
import librosa
import audio

from scipy.io import wavfile
import sounddevice as sd
import time

import librosa.display
import matplotlib.pyplot as plt


def load_wav(path=None):
    if not path:
        path = "/Users/erik/maptaskdataset/maptask/data/dialogues/q1ec1.mix.wav"
    sr, wav = wavfile.read(path)
    f, g = wav.T
    return f, g


def play_audio(wav, sr=20000):
    if not wav.max() < 1:
        wav /= wav.max()*0.99
    sd.default.samplerate = sr
    sd.play(wav)

def wavenet_data():
    out = P.mulaw_quantize(wav, hparams.quantize_channels)
    out8 = P.mulaw_quantize(wav, 256)
    # WAVENENT TRANFSORMATIONS
    # Mu-law quantize

    # Trim silences
    start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
    wav = wav[start:end]
    out = out[start:end]
    constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
    out_dtype = np.int16

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.lws_pad_lr(wav, hparams.fft_size, audio.get_hop_size())

    # zero pad for quantized signal
    out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    timesteps = len(out)

    import matplotlib.pyplot as plt

    plt.subplot(3,1,1)
    specshow(mel_spectrogram.T, sr=20000, hop_length=hparams.hop_size)
    plt.subplot(3,1,2)
    plt.plot(out)
    plt.xlim(0,len(out))
    plt.subplot(3,1,3)
    plt.plot(wav)
    plt.xlim(0,len(wav))
    plt.show()

    out /= out.max()


class MaptaskAcousticSource(FileDataSource):
    def __init__(self, wav_path):
        self.wav_path = wav_path
        # Dont exactly know what these parameters do
        self.sample_rate = 20000
        self.order = 59
        self.frame_period = 5
        self.windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),]

    def collect_files(self):
        wav_paths = sorted(glob(join(self.wav_path, '*.wav')))
        return wav_paths

    def collect_features(self, wav_path):
        '''
        Args:
            wav_path: str
                - path to wav files

        Returns:
            x: np.ndarray (T,)   - time domain audio signal
            mgc: np.ndarray   - time domain audio signal
        '''
        fs, x = wavfile.read(wav_path)
        g, f = x.T
        x = g[:fs*8].astype(np.float64)

        f0, timeaxis = pyworld.dio(x, fs, frame_period=self.frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

        bap = pyworld.code_aperiodicity(aperiodicity, fs)
        mgc = pysptk.sp2mc(spectrogram, order=self.order,
                           alpha=pysptk.util.mcepalpha(fs))
        f0 = f0[:, None]
        lf0 = f0.copy()
        nonzero_indices = np.nonzero(f0)
        lf0[nonzero_indices] = np.log(f0[nonzero_indices])
        vuv = (lf0 != 0).astype(np.float32)

        mgc = apply_delta_windows(mgc, self.windows)
        lf0 = apply_delta_windows(lf0, self.windows)
        bap = apply_delta_windows(bap, self.windows)

        features = np.hstack((mgc, lf0, vuv, bap))

        return x, mgc, lf0, f0, bap, vuv, fs, timeaxis


def vizualize_hardcoded(x, mgc, lf0, f0, vuv, fs, timeaxis):
    plt.subplot(5,1,1)
    plt.plot(x, label="Wav")
    plt.xlim(0,len(x))
    # Spec
    plt.subplot(5,1,2)
    sp = pysptk.mc2sp(mgc[:,:60], alpha=alpha, fftlen=fftlen)
    logsp = np.log(sp)
    librosa.display.specshow(logsp.T, sr=fs, hop_length=hop_length, x_axis="time", y_axis="linear")
    # Lof_f0, Vuv
    plt.subplot(5,1,3)
    # plt.plot(np.exp(lf0[:,0]), linewidth=2, label="Continuous log-f0")
    plt.plot(f0, linewidth=2, label="Continuous log-f0")
    plt.xlim(0,len(f0))
    plt.subplot(5,1,4)
    plt.plot(vuv, linewidth=2, label="Voiced/unvoiced flag")
    plt.xlim(0,len(vuv))
    plt.legend(prop={"size": 14}, loc="upper right")
    # aperiodicity 
    plt.subplot(5,1,5)
    bap = bap[:,:2]
    bap = np.ascontiguousarray(bap).astype(np.float64)
    aperiodicity = pyworld.decode_aperiodicity(bap, fs, fftlen)
    librosa.display.specshow(aperiodicity.T, sr=fs, hop_length=hop_length, x_axis="time", y_axis="linear")
    plt.show()


def print_shaped(x, mgc, lf0, f0, bap, vuv, fs, timeaxis):
    print('x shape: ', x.shape)
    print('mgc shape: ', mgc.shape)
    print('lf0 shape: ', lf0.shape)
    print('f0 shape: ', f0.shape)
    print('bap shape: ', bap.shape)
    print('vuv shape: ', vuv.shape)
    print('fs: ', fs)
    print('len(timeaxis): ', len(timeaxis))


if __name__ == "__main__":

    # Settings
    hparams.sample_rate = 20000
    sr = 20000
    duration = 8
    alpha = 0.41
    fftlen = 1024
    hop_length = 100

    wav_path = "/Users/erik/maptaskdataset/maptask/data/dialogues"
    acoustic_dset = MaptaskAcousticSource(wav_path)
    wavs = acoustic_dset.collect_files()
    x, mgc, lf0, f0, bap, vuv, fs, timeaxis = acoustic_dset.collect_features(wavs[0])

    print_shapes(x, mgc, lf0, f0, bap, vuv, fs, timeaxis)
    vizualize_hardcoded(x, mgc, lf0, f0, bap, vuv, fs, timeaxis)

    fs, x = wavfile.read(wavs[0])
    g, f = x.T
    x = g[:fs*8].astype(np.float64)

    fs = sr
    fft_len = 1024
    hop_length = 256
    frame_period = hop_length / sr * 1000  # hop_length in ms
    f0_floor = 71.  # default
    f0_ceil = 800.  # default

    f0, timeaxis = pyworld.dio(x,
                               fs=sr,
                               f0_floor=f0_floor,
                               f0_ceil=f0_ceil,
                               frame_period=frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    sp = pyworld.cheaptrick(x, f0, timeaxis, fs, fft_size=fft_len)  # Spectrogram
    ap = pyworld.d4c(x, f0, timeaxis, fs, fft_size=fft_len)  # Aperiodicity

    plt.subplot(3,1,1)
    plt.plot(f0)
    plt.subplot(3,1,2)
    plt.plot(lf0)
    plt.subplot(3,1,3)
    librosa.display(sp.T, sr=sr, hop_length=hop_length, y_axis='linear')
    plt.show()

    y = pyworld.synthesize(f0, sp, ap, fs, frame_period)

    play_audio(y)

    bap = pyworld.code_aperiodicity(aperiodicity, fs)
    mgc = pysptk.sp2mc(spectrogram, order=self.order,
                       alpha=pysptk.util.mcepalpha(fs))
    f0 = f0[:, None]
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    vuv = (lf0 != 0).astype(np.float32)
    lf0 = interp1d(lf0, kind="slinear")

    mgc = apply_delta_windows(mgc, self.windows)
    lf0 = apply_delta_windows(lf0, self.windows)
    bap = apply_delta_windows(bap, self.windows)
