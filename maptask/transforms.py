import os
from glob import glob
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import pysptk
import librosa
import parselmouth
from librosa.display import specshow, waveplot
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class MelFeatures(object):
    ''' Transformation of a audio intensity sample array using Librosa '''
    def __init__(self,
                 sr=20000,
                 fft_length=1024,
                 hop_length=256,
                 fmax=8000,
                 n_mels=80,
                 n_mfcc=20,
                 power=2,
                 norm_mfcc=False,
                 use_db=True,
                 use_power=False):
        self.sr = sr
        self.fft_length = fft_length
        self.hop_length = hop_length
        self.fmax = fmax
        self.power = power  # 1: energy, 2:power
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.norm_mfcc = norm_mfcc
        self.use_db = use_db
        self.use_power = use_power

    def standardize_mfcc(self, mfcc):
        ''' Standardize the mfcc values over time mfcc (T, D) '''
        mean = mfcc.mean(0) # (D,)
        std = mfcc.std(0)   # (D,)
        standard = (mfcc - mean) / std
        return standard

    def __call__(self, x):
        mel_power = librosa.feature.melspectrogram(x, sr=self.sr,
                                                   n_fft=self.fft_length,
                                                   hop_length=self.hop_length,
                                                   power=self.power,
                                                   n_mels=self.n_mels,
                                                   fmax=self.fmax)
        mel_db = librosa.power_to_db(mel_power)
        mfcc = librosa.feature.mfcc(S=mel_db, sr=self.sr, n_mfcc=self.n_mfcc)  # S: log-power mel spectrogram
        if self.norm_mfcc:
            mfcc = self.standardize_mfcc(mfcc)
        return mel_db.T, mfcc.T  # (T, n_mels), (T, n_mfcc), T: time_steps/frames


class PitchIntensityPraat(object):
    def __init__(self,
                 sr=20000,
                 hop_length=256,
                 pitch_floor=60.,
                 pitch_ceiling=400.):
        self.sr = sr
        self.hop_length = hop_length
        self.time_step = hop_length/sr
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling

    def __call__(self, x):
        snd = parselmouth.Sound(values=x, sampling_frequency=self.sr)
        pitch = snd.to_pitch(time_step=self.time_step,
                             pitch_floor=self.pitch_floor,
                             pitch_ceiling=self.pitch_ceiling).selected_array['frequency']
        intensity = snd.to_intensity(time_step=self.time_step).values.squeeze()
        return pitch, intensity

class PitchF0PySPTK(object):
    def __init__(self,
                 sr=20000,
                 fft_length=1024,
                 hop_length=256,
                 f0_floor=60.,
                 f0_ceil=240.):
        self.sr = sr
        self.fft_length = fft_length
        self.hop_length = hop_length
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil

    def __call__(self, x):
        pitch = pysptk.swipe(x, fs=self.sr, hopsize=self.hop_length,
                             min=self.f0_floor, max=self.f0_ceil, otype="pitch")
        f0 = pysptk.swipe(x, fs=self.sr, hopsize=self.hop_length,
                          min=self.f0_floor, max=self.f0_ceil, otype="f0")
        return pitch, f0


class F0PyWorld(object):
    def __init__(self,
                 sr=20000,
                 fft_length=1024,
                 hop_length=256,
                 f0_floor=60.,
                 f0_ceil=240.):
        self.sr = sr
        self.fft_length = fft_length
        self.hop_length = hop_length
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil

    def __call__(self, x):
        f0, timeaxis = pyworld.dio(x, fs=self.sr, f0_floor=self.f0_floor, f0_ceil=self.f0_ceil, frame_period=self.frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, self.sr)
        return pitch, f0


class SpeechFeatures(object):
    def __init__(self,
                 sr=20000,
                 fft_length=1024,
                 hop_length=256,
                 frame_period=12.8,
                 alpha=0.441,
                 order=127,
                 f0_floor=71.,
                 f0_ceil=800.,
                 n_mfcc=20,
                 norm_mfcc=True,
                 use_mel=False,
                 bc_threshold=0.33):
        self.sr = sr
        self.fft_length = fft_length
        self.hop_length = hop_length
        self.frame_period = frame_period
        self.alpha = alpha
        self.order = order
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.n_mfcc = n_mfcc
        self.norm_mfcc = norm_mfcc
        self.use_mel = use_mel
        self.bc_threshold = bc_threshold

    def standardize_mfcc(self, mfcc):
        ''' Standardize the mfcc values over time
        mfcc (T, D)
        '''
        mean = mfcc.mean(0) # (D,)
        std = mfcc.std(0)   # (D,)
        standard = (mfcc - mean) / std
        return standard

    def samples_to_frames(self, x):
        pad = len(x) % self.hop_length
        if pad:
            x = np.hstack((np.zeros(pad), x, np.zeros(self.hop_length)))
        frames = librosa.util.frame(x, frame_length=self.hop_length,
                                    hop_length=self.hop_length)
        return frames.T

    def backchannel_activation_to_frames(self, bc):
        bc = self.samples_to_frames(bc)
        frames = bc.shape[0]
        bc_new = []
        for b in bc:
            if b.sum() > int(frames*self.bc_threshold):
                bc_new.append(1.)
            else:
                bc_new.append(0.)
        return np.array(bc_new)

    def process_audio(self, x):
        pitch = pysptk.swipe(x, fs=self.sr, hopsize=self.hop_length, min=self.f0_floor, max=self.f0_ceil, otype="pitch")

        f0, timeaxis = pyworld.dio(x, fs=self.sr, f0_floor=self.f0_floor, f0_ceil=self.f0_ceil, frame_period=self.frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, self.sr)

        # x_frame = self.samples_to_frames(x)

        if self.use_mel:
            mel = librosa.feature.melspectrogram(x, sr=self.sr, n_fft=self.fft_length, hop_length=self.hop_length)
            mfcc = librosa.feature.mfcc(S=mel, sr=self.sr, n_mfcc=self.n_mfcc)
            if self.norm_mfcc:
                mfcc = self.standardize_mfcc(mfcc)
            return {'mel': mel.T, 'mfcc': mfcc.T, 'f0': f0, 'pitch': pitch}
        else:
            ap = pyworld.d4c(x, f0, timeaxis, self.sr, fft_size=self.fft_length)  # Aperiodicity
            sp = pyworld.cheaptrick(x, f0, timeaxis, self.sr,
                                    fft_size=self.fft_length)
            return {'sp': sp, 'ap': ap, 'f0': f0, 'pitch': pitch}

    def __call__(self, sample):
        context, backchannel, bc_class = sample
        context = self.process_audio(context)
        backchannel = self.process_audio(backchannel)
        bc_class = self.backchannel_activation_to_frames(bc_class)
        return context, backchannel, bc_class


if __name__ == "__main__":
    # TODO
    # Make ipynb comparing extraction methods
    # Plot f0 & pitch
    # Load Data
    # testpath = "/Users/erik/maptaskdataset/maptask/data/dialogues/q1ec1.mix.wav"
    testpath = "/home/erik/Audio/maptaskdataset/maptask/data/dialogues/q1ec1.mix.wav"
    sr, wav = wavfile.read(testpath)
    duration = 8
    wav = wav[:sr*duration]
    x = np.ascontiguousarray(wav[:,0]).astype(np.float64)
    sr = 20000
    fs = sr
    n_fft = 1024
    frame_length = 1024
    hop_length = 256
    n_mels = 80
    n_mfcc = 40
    fmax = 8000  # mfcc
    f0_floor = 60.
    f0_ceil = 800.
    frame_period = 12.8

    snd = parselmouth.Sound(values=x, sampling_frequency=sr)

    # Transformers
    pitch_intensity_praat = PitchIntensityPraat()
    melspec_mfcc = MelFeatures()
    pitch_f0 = PitchF0PySPTK()

    mel, mfcc = melspec_mfcc(x)
    pitch, intensity = pitch_intensity_praat(x)
    p, f0 = pitch_f0(x)

    # Rms and intensity ?
    rms = librosa.feature.rmse(x, frame_length=frame_length,
                               hop_length=hop_length).squeeze()
    L = 10 * np.log10(rms)
    rms_min = rms.min()
    rms = rms/rms.min() + rms.min()

    print('pitch: ', pitch.shape)
    print('intensity: ', intensity.shape)
    print('p: ', p.shape)
    print('f0: ', f0.shape)
    print('rms: ', rms.shape)
    print('mel: ', mel.shape)
    print('mfcc: ', mfcc.shape)

    plt.suptitle('data')
    plt.subplot(2,1,1)
    plt.plot(pitch, label='Pitch')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(intensity, label='intensity')
    plt.legend()

    plt.subplot(6,1,3)
    plt.plot(rms, label='RMS')
    plt.plot([300]*rms.size)
    plt.legend()
    plt.xticks([])
    plt.subplot(6,1,4)
    plt.plot(L, label='Log-RMS')
    plt.legend()
    plt.xticks([])
    plt.subplot(6,1,5)
    plt.plot(f0, label='F0')
    plt.legend()
    plt.xticks([])
    plt.subplot(6,1,6)
    plt.plot(p, label='pitch')
    plt.legend()
    plt.xticks([])
    plt.pause(0.1)


    # Listen
    import sounddevice as sd
    sd.default.samplerate = 20000
    wav = x.astype(np.float)
    wav = x/x.max()
    sd.play(wav)

    Ms = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=Ms, sr=sr, n_mfcc=n_mfcc).T  # S: log-power mel spectrogram
    mel = Ms.T

    specshow(Ms.T, sr=sr, hop_length=hop_length, y_axis='mel')
