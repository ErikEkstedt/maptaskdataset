import os
from glob import glob
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import torch

import pysptk
import librosa
from librosa.display import specshow, waveplot
from torch.utils.data import Dataset, DataLoader
from maptask.transforms import PitchIntensityPraat, MelFeatures


# Transforms
class PitchIntensityDataset(Dataset):
    def __init__(self,
                 root_dir,
                 sr=20000,
                 fft_length=1024,
                 hop_length=256,
                 frame_period=12.8,
                 pitch_floor=71.,
                 pitch_ceiling=800.,
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
        self.frame_period = frame_period
        self.pitch_floor = pitch_floor
        self.pitch_ceiling= pitch_ceiling
        self.bc_threshold = bc_threshold
        self.transform = PitchIntensityPraat(sr=sr,
                                             hop_length=hop_length,
                                             pitch_floor=pitch_floor,
                                             pitch_ceiling=pitch_ceiling)

    def get_data_dims(self):
        return {'bc_size': self.bc_size, 'context_size': self.context_size}

    def __len__(self):
        return len(self.data)

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

    def __getitem__(self, idx):
        fpath = os.path.join(self.root_dir, self.data[idx])
        # sample = np.load(fpath)
        # context, backchannel, bc_class = sample
        context, backchannel, bc_class = np.load(fpath)

        data = {}
        pitch, intensity = self.transform(context)
        context =  {'pitch': pitch, 'intensity': intensity}

        pitch, intensity = self.transform(backchannel)
        bc_class = self.backchannel_activation_to_frames(bc_class)
        backchannel = {'pitch': pitch, 'intensity': intensity, 'bc_class': bc_class}
        return context, backchannel


class PIDataLoader(DataLoader):
    def __init__(self, dset, batch_size, frames_in, frames_out, verbose=False, *args, **kwargs):
        super().__init__(dset, batch_size, *args, **kwargs)
        self.frames_in = frames_in
        self.frames_out = frames_out
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
            context, backchannel = batch
            batch_size, n_frames = context['pitch'].shape  # arbitrary choice
            print('Batch_size: ', batch_size)
            print('n_frames: ', n_frames)

            reset = True
            # start: 0 -> 157 - 1
            for start in range(n_frames - self.frames_in + 1 -
                               self.frames_out):
                end = start + self.frames_in
                target_start = end
                target_end = target_start + self.frames_out

                # TODO
                Context = {'pitch': context['pitch'][:, start:end],
                           'intensity': context['intensity'][:, start:end]}

                # Backchannel
                BC = {'pitch': backchannel['pitch'][:,target_start:target_end],
                      'intensity': backchannel['intensity'][:,target_start:target_end],
                      'bc_class': backchannel['bc_class'][:,target_start:target_end]}

                yield (Context, BC, reset)
                reset = False

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

## Simplest of dataset. Loads numpy files from memory pitch and intensity
class DSet(Dataset):
    def __init__(self, root_dir='data'):
        self.root_dir = root_dir
        self.pitch = torch.tensor(np.load(os.path.join(root_dir, 'pitch.npy'))[:, :,
                                                                       :-1]).float()
        self.intensity = torch.tensor(np.load(os.path.join(root_dir,
                                                   'intensity.npy'))).float()

    def __len__(self):
        return self.pitch.shape[1]

    def get_random(self):
        return self[np.random.randint(len(self))]

    def __getitem__(self, idx):
        data = torch.stack((self.pitch[0, idx, :-1],
                           self.intensity[0, idx, :-1],
                           self.pitch[1, idx, :-1],
                           self.intensity[1, idx, :-1]))
        target = torch.stack((self.pitch[1, idx, 1:],
                              self.intensity[1, idx, 1:]))
        return data, target


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


def demo(dset):
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


if __name__ == "__main__":
    root_dir = '/Users/erik/maptaskdataset/maptask/data/processed'
    use_mel = False
    print('Use mel: ', use_mel)
    dset = TransformDataset(root_dir, use_mel=use_mel, norm_mfcc=False)
    demo(dset)

    from tqdm import tqdm
    root_dir = '/home/erik/Audio/maptaskdataset/maptask/data/processed'
    dset = PitchIntensityDataset(root_dir)

    c_pitch, c_intensity = [], []
    bc_pitch, bc_intensity, bc_class = [], [], []
    for d in tqdm(dset):
        context, bc = d
        c_pitch.append(context['pitch'])
        c_intensity.append(context['intensity'])
        bc_pitch.append(bc['pitch'])
        bc_intensity.append(bc['intensity'])
        bc_class.append(bc['bc_class'])
        # print('context pitch: ', context['pitch'].shape)
        # print('context intensity: ', context['intensity'].shape)
        # print(bc.keys())
        # print('backchannel pitch: ', bc['pitch'].shape)
        # print('backchannel intensity: ', bc['intensity'].shape)
        # input()

    pitch = np.array([c_pitch, bc_pitch])
    intensity = np.array([c_intensity , bc_intensity])
    p_stand = (pitch - pitch.mean()) / pitch.std()
    i_stand = (intensity - intensity.mean()) / intensity.std()


    pidloader = PIDataLoader(dset,
                           batch_size=128,
                           frames_in=5,
                           frames_out=1,
                           verbose=False,
                           num_workers=4)

    # Todo
    # Save data directly... way too much overhead to extract features on demand
    # ~153 frames in one 2 sec bc-snippet
    # Each frame: 2 values -> 3 out (pitch, intensity, class)
    for d in pidloader:
        context, bc, new_sequence = d
        print(context.keys())
        print('context pitch: ', context['pitch'].shape)
        print('context intensity: ', context['intensity'].shape)
        print(bc.keys())
        print('backchannel pitch: ', bc['pitch'].shape)
        print('backchannel intensity: ', bc['intensity'].shape)
        print(new_sequence)
        input()

    dloader = DataLoader(dset, batch_size=64)
    for d in tqdm(dloader):
        context, bc = d
        print(context.keys())
        print('context pitch: ', context['pitch'].shape)
        print('context intensity: ', context['intensity'].shape)
        print(bc.keys())
        print('backchannel pitch: ', bc['pitch'].shape)
        print('backchannel intensity: ', bc['intensity'].shape)
        print(new_sequence)
        input()





