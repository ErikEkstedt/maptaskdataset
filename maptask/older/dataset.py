import os
from os.path import join, split
import pathlib
from tqdm import tqdm
from scipy.io.wavfile import read, write
import xml.etree.ElementTree as ET
import librosa
from librosa import time_to_samples as librosa_time_to_samples
from librosa import samples_to_frames as librosa_samples_to_frames
import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MEL2 as MelSpectrogram
from torchaudio.transforms import F2M

from maptask.utils import get_paths, load_audio, visualize_datapoint, sound_datapoint


# TODO
# Should do this when downloading dataset
#########################################
# Using scipy read -> Error: q6ec2.mix.wav
# Using torchaudio load -> zeros
# !!! Remove q6ec2, it is CORRUPT !!!
#########################################


# Dataset
class MaptaskDataset(Dataset):
    ''' 128 dialogs, 16-bit samples, 20 kHz sample rate, 2 channels per conversation '''
    def __init__(self,
                 pause=0.5,
                 context=2,
                 max_utterences=1,
                 sample_rate=20000,
                 use_spectrogram=True,
                 window_size=512,
                 hop_length=256,
                 n_fft=None,
                 pad=0,
                 n_mels=40,
                 torch_load_audio=True,
                 audio=None,
                 normalize_audio=True,
                 root=None,
                 n_files=None):
        self.maptask = Maptask(pause, max_utterences, root, n_files)
        self.paths = self.maptask.paths
        self.session_names = self.maptask.session_names

        # Mel
        self.use_spectrogram = use_spectrogram
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.pad = pad
        self.n_mels = n_mels

        self.mel_spec = MelSpectrogram(sr=self.sample_rate,
                                       ws=self.window_size,
                                       hop=self.hop_length,
                                       n_fft=self.n_fft,
                                       pad=self.pad,
                                       n_mels=self.n_mels)
        # Audio
        self.normalize_audio = normalize_audio
        self.context = context
        self.torch_load_audio = torch_load_audio
        if audio:
            self.audio = audio
        else:
            self.audio = load_audio(self.paths['dialog_path'],
                                    self.session_names,
                                    self.torch_load_audio,
                                    self.normalize_audio)

    def __len__(self):
        return len(self.maptask.back_channel_list)

    def griffin_lim(self, magnitude, iters=30):
        '''
        based on:
        https://github.com/soobinseo/Tacotron-pytorch/blob/master/data.py
        in turn based on:
        librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        S_complex = np.abs(magnitude).astype(np.complex)
        y = librosa.istft(S_complex * angles)
        for i in range(iters):
            _, angles = librosa.magphase(librosa.stft(y))
            y = librosa.istft(S_complex * angles)
        return y

    def __getitem__(self, idx):
        bc = self.maptask.back_channel_list[idx]
        start, end = bc['sample']

        n_samples = librosa_time_to_samples(self.context, sr=self.sample_rate)
        context = torch.zeros(n_samples)
        back_channel = torch.zeros(n_samples)

        # Find start of context >= 0
        context_start = end - n_samples
        if context_start < 0:
            context_start = 0

        y = self.audio[bc['name']]  # load correct audio array
        if bc['user'] == 'f':
            # back channel generator is 'f'
            tmp_context = y[context_start:end, 0]
            tmp_back_channel = y[context_start:end, 1]
        else:
            # back channel generator is 'g'
            tmp_context = y[context_start:end, 1]
            tmp_back_channel = y[context_start:end,0]

        context[-tmp_context.shape[0]:] = tmp_context
        back_channel[-tmp_back_channel.shape[0]:] = tmp_back_channel

        n_samples_bc = end - start
        back_channel_class = torch.zeros(back_channel.shape)
        back_channel_class[-n_samples_bc:] = 1

        # Spectrograms
        if self.use_spectrogram:
            context_spec = self.mel_spec(context.unsqueeze(0)).squeeze(0)
            back_channel_spec = self.mel_spec(back_channel.unsqueeze(0)).squeeze(0)

            bc_frames = librosa_samples_to_frames(n_samples_bc, hop_length=self.hop_length)
            back_channel_spec_class = torch.zeros(back_channel_spec.shape[0])
            back_channel_spec_class[-bc_frames:] = 1

            return {'context_audio': context,
                    'context_spec': context_spec,
                    'back_channel_audio': back_channel,
                    'back_channel_spec': back_channel_spec,
                    'back_channel_class': back_channel_class,
                    'back_channel_spec_class': back_channel_spec_class,
                    'back_channel_word': bc['words'][0]}
        else:
            return {'context_audio': context,
                    'back_channel_audio': back_channel,
                    'back_channel_class': back_channel_class,
                    'back_channel_word': bc['words'][0]}


# DataLoaders
class MaptaskAudioDataloader(DataLoader):
    def __init__(self, dset, batch_size, pred, seq_len, overlap_len, *args, **kwargs):
        super().__init__(dset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len
        self.pred = pred

    def __iter__(self):
        for batch in super().__iter__():
            # batch['back_channel_audio'].shape)
            # batch['context_audio'].shape)
            # batch['back_channel_class'].shape)
            (batch_size, n_samples) = batch['back_channel_audio'].shape

            reset = True
            for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
                start = seq_begin - self.overlap_len
                end = seq_begin + self.seq_len

                bc_class = batch['back_channel_class'][:, start:end]
                bc_seq = batch['back_channel_audio'][:, start:end]
                context_seq = batch['context_audio'][:, start:end]

                context = context_seq[:,:-self.pred]
                self_context = bc_seq[:,:-self.pred]
                self_context_class = bc_class[:,:-self.pred]
                target = bc_seq[:, self.overlap_len:].contiguous()
                target_class = bc_class[:, self.overlap_len:]

                yield {'context': context,
                       'self_context': self_context,
                       'self_context_class': self_context_class,
                       'target': target,
                       'target_class': target_class,
                       'reset': reset}
                reset = False


class MaptaskSpecDataloader(DataLoader):
    def __init__(self, dset, batch_size, pred, seq_len, overlap_len, *args, **kwargs):
        super().__init__(dset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len
        self.pred = pred

    def __iter__(self):
        for batch in super().__iter__():
            # batch['context_spec'].shape
            # batch['back_channel_spec'].shape
            # batch['back_channel_spec_class'].shape
            (batch_size, n_frames, features) = batch['back_channel_spec'].shape

            reset = True
            for seq_begin in range(self.overlap_len, n_frames, self.seq_len):
                start = seq_begin - self.overlap_len
                end = seq_begin + self.seq_len

                bc_seq = batch['back_channel_spec'][:, start:end]
                bc_class = batch['back_channel_spec_class'][:, start:end]
                context_seq = batch['context_spec'][:, start:end]

                input_seq = (context_seq[:,:-self.pred], bc_seq[:,:-self.pred])
                target_seq = bc_seq[:, self.overlap_len:].contiguous()
                target_class = bc_class[:, self.overlap_len:].contiguous()
                yield (input_seq, reset, target_seq, target_class)
                reset = False


def get_dataset_dataloader(pause=0.5,
                           context=2,
                           max_utterences=1,
                           sample_rate=20000,
                           use_spectrogram=True,
                           window_size=512,
                           hop_length=256,
                           n_fft=None,
                           pad=0,
                           n_mels=40,
                           torch_load_audio=True,
                           audio=None,
                           normalize_audio=True,
                           root=None,
                           batch_size=64,
                           pred=1,
                           seq_len=10,
                           overlap_len=2,
                           n_files=None):

    dset = MaptaskDataset(pause,
                          context,
                          max_utterences,
                          sample_rate,
                          use_spectrogram,
                          window_size,
                          hop_length,
                          n_fft,
                          pad,
                          n_mels,
                          torch_load_audio,
                          audio,
                          normalize_audio,
                          root,
                          n_files)

    if use_spectrogram:
        dloader = MaptaskSpecDataloader(dset, batch_size, pred, seq_len, overlap_len)
    else:
        dloader = MaptaskAudioDataloader(dset, batch_size, pred, seq_len,
                                         overlap_len, drop_last=True)
    return dset, dloader


# Loads Numpy audio
class MaptaskDataset2(Dataset):
    def __init__(self,
                 sample_rate=20000,
                 use_spectrogram=False,
                 window_size=512,
                 hop_length=256,
                 n_fft=None,
                 pad=0,
                 n_mels=40,
                 root='data/chopped',
                 n_files=None):
        self.root = root
        self.files = os.listdir(root)

        # Mel
        self.use_spectrogram = use_spectrogram
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.pad = pad
        self.n_mels = n_mels

        self.mel_spec = MelSpectrogram(sr=self.sample_rate,
                                       ws=self.window_size,
                                       hop=self.hop_length,
                                       n_fft=self.n_fft,
                                       pad=self.pad,
                                       n_mels=self.n_mels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = join(self.root, self.files[idx])
        data = np.load(fpath)
        context, backchannel, bc_class = data
        context /= data.max()
        backchannel /= data.max()
        return {'context': torch.tensor(context).float(),
                'backchannel': torch.tensor(backchannel).float(),
                'bc_class': torch.tensor(bc_class).float()}


class MaptaskDataloader(DataLoader):
    def __init__(self, dset, batch_size, seq_len, hop_len, prediction,  *args, **kwargs):
        super().__init__(dset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.hop_len = hop_len
        self.prediction = prediction

    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch['context'].shape

            reset = True
            for start in range(n_samples-self.seq_len-self.prediction):
                end = start + self.seq_len
                target_start = start + self.prediction
                target_end = end + self.prediction

                context = batch['context'][:, start:end]
                bc = batch['backchannel'][:, start:end]
                bc_class = batch['bc_class'][:, start:end]

                target = batch['backchannel'][:, target_start:target_end]
                target_class = batch['bc_class'][:, target_start:target_end]

                yield {'context': context, 'backchannel': bc, 'bc_class':
                       bc_class, 'target': target, 'target_class': target_class}
                reset = False


if __name__ == "__main__":
    # dset = MaptaskDataset2()
    # dloader = MaptaskDataloader(dset, batch_size=32, seq_len=10, hop_len=5,
    #                             prediction=1)

    # for batch in dloader:
    #     print(batch['context'].shape)
    #     print(batch['backchannel'].shape)
    #     print(batch['bc_class'].shape)
    #     print(batch['target'].shape)
    #     print(batch['target_class'].shape)
    #     input()

    # import sounddevice as sd
    # sd.default.samplerate=20000

    # sd.play(np.stack((context, backchannel), axis=1))
    # sd.play(bc_class)

    # print('PyTorch datasets & loader')
    # print('Creating Dataset and loading audio')
    # dset = MaptaskDataset(pause=0.5, max_utterences=1)
    # audio = dset.audio

    # # for reusing audio in REPL
    # dset = MaptaskDataset(pause=0.5, max_utterences=1, context=2, audio=audio)

    # # ---------------------------------------------------------------------
    # print('Audio: DataLoader')
    # maploader = MaptaskAudioDataloader(dset, pred=1, seq_len=300, overlap_len=100, batch_size=128, num_workers=4)
    # for batch in maploader:
    #     print(type(batch))
    #     print(batch[0][0].shape)
    #     print(batch[0][1].shape)
    #     print(batch[1])
    #     print(batch[2].shape)
    #     ans = input('Press n for quit')
    #     if ans == 'n':
    #         break

    # # ---------------------------------------------------------------------
    # print('Spectrogram: DataLoader')
    # maploader = MaptaskSpecDataloader(dset, pred=1, seq_len=10, overlap_len=2, batch_size=128, num_workers=4)
    # for batch in maploader:
    #     print(type(batch))
    #     print(batch[0][0].shape)
    #     print(batch[0][1].shape)
    #     print(batch[1])
    #     print(batch[2].shape)
    #     print(batch[3].shape)
    #     ans = input('Press n for quit')
    #     if ans == 'n':
    #         break

    # # ---------------------------------------------------------------------

    # print('Listen to audio datapoints')
    # while True:
    #     idx = int(torch.randint(len(dset), (1,)).item())
    #     output = dset[idx]
    #     visualize_datapoint(output)
    #     sound_datapoint(output)
    #     ans = input('Press n for quit')
    #     if ans == 'n':
    #         break
