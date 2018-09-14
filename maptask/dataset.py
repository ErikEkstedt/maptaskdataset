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


# Transform
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


# Dataset
class TransformDataset(Dataset):
    def __init__(self,
                 root_dir,
                 sr=20000,
                 context=2,
                 fft_length=1024,
                 hop_length=256,
                 frame_period=12.8,
                 alpha=0.441,
                 order=40,
                 f0_floor=71.,
                 f0_ceil=800.,
                 n_mfcc=20,
                 norm_mfcc=True,
                 use_mel=False,
                 n_mels=128,
                 bc_threshold=0.33):
        """
        Args:
        root_dir (string): Directory with all the wav.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.root_dir = root_dir
        self.data = [f for f in glob(os.path.join(root_dir, '*.npy'))]

        # Transform
        self.sr = sr
        self.fft_length = fft_length
        self.hop_length = hop_length
        self.ap_size = fft_length // 2 + 1  # 1024 -> 513
        self.sp_size = fft_length // 2 + 1  # 1024 -> 513
        self.n_mfcc = n_mfcc
        self.norm_mfcc = norm_mfcc
        self.use_mel = use_mel
        self.n_mels = n_mels
        self.frame_period = frame_period
        self.alpha = alpha
        self.order = order
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.bc_threshold = bc_threshold
        self.transform = SpeechFeatures(sr,
                                        fft_length,
                                        hop_length,
                                        frame_period,
                                        alpha,
                                        order,
                                        f0_floor,
                                        f0_ceil,
                                        n_mfcc,
                                        norm_mfcc,
                                        use_mel,
                                        bc_threshold)
        # Dimensions
        self.n_frames = sr*context // hop_length + 1  # 157
        if use_mel:
            # context = n_mels + n_mfcc + n_ap_channels + f0 + pitch
            # bc = n_mels + n_mfcc + n_ap_channels + f0 + pitch + bc_class
            # => 128 + 20 + 1 + 1 + 1 = 
            self.bc_size = self.n_mels + self.n_mfcc + 1 + 1 + 1
            self.context_size = self.n_mels + self.n_mfcc + 1 + 1
            # times 2 channels (speaker, backchanneler)
            # => 663 * 2 = 1326
            self.utterence_size = 2*self.frame_out_size*self.n_frames
        else:
            # context = sp + ap + f0 + pitch
            # bc = sp + ap + f0 + pitch + bc_class
            # => 513 + 513 + 1 + 1 = 1028
            self.bc_size = self.sp_size + self.ap_size + 1 + 1 + 1
            self.context_size = self.sp_size + self.ap_size + 1 + 1
            # times 2 channels (speaker, backchanneler)
            # => 663 * 2 = 1326

    def get_data_dims(self):
        return {'bc_size': self.bc_size, 'context_size': self.context_size}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fpath = os.path.join(self.root_dir, self.data[idx])
        sample = np.load(fpath)
        context, backchannel, bc_class = sample
        # print('idx: ', idx)

        if self.transform:
            sample = self.transform(sample)
            # {'audio_frames': x, 'mel': mel, 'mfcc': mfcc, 'ap': ap, 'f0': f0, 'pitch': pitch}
        return sample


# DataLoader
class TranformDataloader(DataLoader):
    def __init__(self, dset, batch_size, frames_in, frames_out, verbose=False, *args, **kwargs):
        super().__init__(dset, batch_size, *args, **kwargs)
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.use_mel = dset.use_mel
        self.verbose = verbose

    def get_data_dims(self):
        dset_dims = self.dataset.get_data_dims()
        bc_size = dset_dims['bc_size']
        context_size = dset_dims['context_size']

        # Frames in/out
        context_size = (self.frames_in, context_size)
        bc_size = (self.frames_out, bc_size)
        return {'bc_size': bc_size, 'context_size': context_size}

    def __iter__(self):
        for batch in super().__iter__():
            context, backchannel, bc_class = batch
            batch_size, n_frames = context['f0'].shape  # arbitrary choice
            print('Batch_size: ', batch_size)
            print('n_frames: ', n_frames)

            if self.use_mel:

                if self.verbose:
                    print('mel shape: ', context['mel'].shape)
                    print('mfcc shape: ', context['mfcc'].shape)
                    print('f0 shape: ', context['f0'].shape)
                    print('pitch shape: ', context['pitch'].shape)
                    print('bc_class', bc_class.shape)
                    input()

                reset = True
                # start: 0 -> 157 - 1
                for start in range(n_frames - self.frames_in + 1 -
                                   self.frames_out):
                    end = start + self.frames_in
                    target_start = end
                    target_end = target_start + self.frames_out

                    Context = {'mel': context['mel'][:, start:end, :],
                          'mfcc': context['mfcc'][:, start:end, :],
                          'f0': context['f0'][:, start:end],
                          'pitch': context['pitch'][:, start:end]}

                    # Backchannel
                    BC = {'mel': backchannel['mel'][:, target_start:target_end, :],
                          'mfcc': backchannel['mfcc'][:, target_start:target_end, :],
                          'f0': backchannel['f0'][:, target_start:target_end],
                          'pitch': backchannel['pitch'][:, target_start:target_end],
                          'bc_class': bc_class[:,target_start:target_end]}

                    yield (Context, BC, reset)
                    reset = False

            else:
                if self.verbose:
                    print('sp shape: ', context['sp'].shape)
                    print('ap shape: ', context['ap'].shape)
                    print('f0 shape: ', context['f0'].shape)
                    print('pitch shape: ', context['pitch'].shape)
                    print('bc_class', bc_class.shape)
                    input()

                reset = True
                # start: 0 -> 157 - 1
                for start in range(n_frames - self.frames_in + 1 -
                                   self.frames_out):
                    end = start + self.frames_in
                    target_start = end
                    target_end = target_start + self.frames_out

                    Context = {'sp': context['sp'][:, start:end, :],
                          'ap': context['ap'][:, start:end, :],
                          'f0': context['f0'][:, start:end],
                          'pitch': context['pitch'][:, start:end]}

                    # Backchannel
                    BC = {'sp': backchannel['sp'][:, target_start:target_end, :],
                          'ap': backchannel['ap'][:, target_start:target_end, :],
                          'f0': backchannel['f0'][:, target_start:target_end],
                          'pitch': backchannel['pitch'][:, target_start:target_end],
                          'bc_class': bc_class[:,target_start:target_end]}

                    yield (Context, BC, reset)
                    reset = False


# Visualizers
def print_sample_example(dset):
    print('frame size: ', dset.frame_out_size )
    print('frames: ', dset.n_frames)
    print('utterence size: ', dset.utterence_size)
    print()
    context, backchannel, bc_class = dset[0]
    print()
    print('Context')
    print('-'*55)
    for k, v in context.items():
        print('{}: {}'.format(k, v.shape))
    print()
    print('backchannel')
    print('-'*55)
    for k, v in backchannel.items():
        print('{}: {}'.format(k, v.shape))
    print()
    print('-'*55)
    print('bc_class: ', bc_class.shape)


def plot_sample_example(dset, datatype='context', idx=None):
    def plot_sample(data, datatype='context'):
        plt.figure(datatype)
        i = 1
        n = 5 if datatype == 'backchannel' else 4
        if dset.use_mel:
            plt.subplot(n,1,i); i+=1
            plt.title('Mel Spectrogram')
            specshow(np.log(data['mel']).T, sr=dset.sr, hop_length=dset.hop_length,
                     y_axis='mel', cmap='magma')
            plt.subplot(n,1,i); i+=1
            plt.title('MFCC')
            specshow(data['mfcc'].T, sr=dset.sr, hop_length=dset.hop_length, cmap='magma')
            # plt.plot([m for m in data['mfcc'].T])
        else:
            plt.subplot(n,1,i); i+=1
            plt.title('Spectrogram')
            # specshow(np.log(data['sp']).T, sr=dset.sr, hop_length=dset.hop_length,
            #          y_axis='linear', cmap='magma')
            sp = data['sp'] / data['sp'].max()
            # sp *= 80
            specshow(np.log(sp).T, sr=dset.sr, hop_length=dset.hop_length,
                     y_axis='linear', cmap='magma')
            plt.subplot(n,1,i); i+=1
            plt.title('Aperiodicity')
            specshow(data['ap'].T, sr=dset.sr, hop_length=dset.hop_length,
                     y_axis='linear', cmap='magma')
        plt.subplot(n,1,i); i+=1
        plt.title('F0')
        plt.plot(data['f0'])
        plt.xlim(0, len(data['f0']))
        plt.xticks([])
        plt.subplot(n,1,i); i+=1
        plt.title('Pitch')
        plt.plot(data['pitch'])
        plt.xlim(0, len(data['pitch']))
        if datatype=='backchannel':
            plt.xticks([])
            plt.subplot(n,1,i); i+=1
            plt.plot(bc_class)
            plt.xlim(0, len(bc_class))
            plt.title('backchannel activation')
    if not idx:
        idx = np.random.randint(len(dset))
    context, backchannel, bc_class = dset[idx]
    plot_sample(context)
    plt.tight_layout()
    plt.show()
    plt.close()
    plot_sample(backchannel, datatype='backchannel')
    plt.tight_layout()
    plt.show()
    plt.close()


def synthesize_audio(dset):
    import sounddevice as sd
    sd.default.samplerate = dset.sr
    for f in range(10):
        context, bc, bc_class = dset[np.random.randint(len(dset))]
        y = pyworld.synthesize(context['f0'],
                               context['sp'],
                               context['ap'],
                               dset.sr,
                               dset.frame_period)
        y = y/y.max() * 0.91
        sd.play(y)
        input()


if __name__ == "__main__":
    root_dir = '/Users/erik/maptaskdataset/maptask/data/processed'

    # Dataset with transform
    use_mel = False
    print('Use mel: ', use_mel)
    dset = TransformDataset(root_dir, use_mel=use_mel, norm_mfcc=False)
    dloader = TranformDataloader(dset,
                                 batch_size=3,
                                 frames_in=5,
                                 frames_out=1,
                                 verbose=False,
                                 num_workers=4)

    ans = input('Iterate through dataloader? (y/n)')
    if ans == 'y' or ans == 'Y':
        for i, batch in enumerate(dloader):
            context, bc, RESET_RNN = batch
            print('Mini-batch: ', i)
            if RESET_RNN:
                print('RNN reset')
            print('\nInput\t Target')
            for (kd,vk), (kt, vt) in zip(context.items(), bc.items()):
                print('{} : {}\t {}'.format(kd, vk.shape, vt.shape))
            ans = input('Press Enter to continue: ')
            if ans:
                break

    # Visualize
    ans = input('Visualize samples? (y/n)')
    if ans == 'y' or ans == 'Y':
        # print_sample_example(dset)
        while True:
            plot_sample_example(dset)
            ans = input('Press any key except Enter to stop')
            if ans:
                break


    # Synthesize Audio
    if not dset.use_mel:
        synthesize_audio(dset, batch_size=32, frames_in=1, frames_out=1, num_workers=2)

    import sys
    sys.exit(0)

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
