import os
from glob import glob
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import pysptk
import pyworld
import librosa
from librosa.display import specshow, waveplot
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    # TODO
    # Make ipynb comparing extraction methods
    # Plot f0 & pitch
    # Load Data

    testpath = "/Users/erik/maptaskdataset/maptask/data/dialogues/q1ec1.mix.wav"
    sr, wav = wavfile.read(testpath)
    duration = 8
    wav = wav[:sr*duration]
    x = wav[:sr*duration, 0]
    x = np.ascontiguousarray(wav[:16000, 0]).astype(np.float64)

    pysptk_pitch = pysptk.swipe(x[:2000], fs=sr, hopsize=hop_length,
                                min=f0_floor, max=f0_ceil, otype="pitch")
    pyspkt_f0_swipe = pysptk.swipe(x, fs=sr, hopsize=hop_length,
                                   min=f0_floor, max=f0_ceil, otype="f0")
    pyspkt_f0_rapt = pysptk.rapt(x.astype(np.float32), fs=sr, hopsize=hop_length,
                                 min=f0_floor, max=f0_ceil, otype="f0")
    f0, timeaxis = pyworld.dio(x, fs=sr, f0_floor=f0_floor,
                               f0_ceil=f0_ceil, frame_period=frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)



    # Spectrogram: magnitude**2
    sp = pyworld.cheaptrick(x, f0, timeaxis, fs, fft_size=fft_length)  
    mel_spec = pysptk.sp2mc(sp, order=order, alpha=alpha)
    ap = pyworld.d4c(x, f0, timeaxis, fs, fft_size=fft_length)  # Aperiodicity
    mel_spec_audio = audio.melspectrogram(x)
    mfcc = pysptk.mfcc(mel_spec, fs=sr, alpha=alpha, order=80, num_filterbanks=100)
    mfcc = standardize_mfcc(mfcc)
    energy = pysptk.mc2e(mel_spec, alpha=alpha)
