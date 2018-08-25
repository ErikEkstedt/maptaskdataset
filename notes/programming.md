# Programming

## Utils
```python
import os
from os.path import join
from tqdm import tqdm
from scipy.io.wavfile import read, write
import xml.etree.ElementTree as ET
from librosa import time_to_samples as librosa_time_to_samples

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MEL2 as MelSpectrogram
from torchaudio.transforms import SPECTROGRAM as Spec

```

```python
def get_paths(root_path=None):
    # Assumes maptask dialogue files are downloaded in $dataPath
    if not root_path:
        try:
            full_path = os.path.realpath(__file__)
            root_path, filename = os.path.split(full_path)
        except: # for ipython repl error
            print('Assumes this repo is in home directory')
            root_path = join(os.path.expanduser('~'), 'maptaskdataset')
    data_path = os.path.realpath(join(root_path, 'data'))
    return {'data_path' : data_path,
            'annotation_path' : join(data_path, 'maptaskv2-1'),
            'dialog_path' : join(data_path, 'dialogues'),
            'mono_path' : join(data_path, 'dialogues_mono'),
            'timed_units_path': join(data_path, "maptaskv2-1/Data/timed-units"),
            'gemap_path' : join(data_path, 'gemaps'),
            'opensmile_path' : join(os.path.expanduser('~'), 'opensmile-2.3.0')}
```


### Paths
Useful script to load a custom data path or generic which I try to use
across systems.

This one is helpful and could be booilerplate in all code.
A function that could use a custom path or go to standard.

## Maptask
This class iterates through the annotations provided by maptask and extracts
backchannel uttterences.

The annotations in the dataset contains words spoken and correlating timings.
Utterences from one speaker seperated by less than `pause` are combined to
make a longer utterence, a sentence.

All utterences which only contains one word are extracted and sorted by
popularity. From this list of one word utterences the most common ones which
are known to be back channels are then extracted.

As of now this means that the 6 words below make up all utterences in the
dataset:

  self.back_channels = ['right', 'okay', 'mmhmm', 'uh-huh', 'yeah', 'mm']

This class stores all extracted backchannels in a list where each entry in
the list is a dict:

	{'name': name, 'time', time, 'sample': sample, 'words': words}

Here 'name' is the name of the session, 'time' and 'sample' are the timings of
the utterence and 'words' is a list of strings.

```python
class Maptask(object):
    '''
    This class iterates through the annotations provided by maptask and extracts
    backchannel uttterences.

    The annotations in the dataset contains words spoken and correlating timings.
    Utterences from one speaker seperated by less than `pause` are combined to
    make a longer utterence, a sentence.

    All utterences which only contains one word are extracted and sorted by
    popularity. From this list of one word utterences the most common ones which
    are known to be back channels are then extracted.

    As of now this means that the 6 words below make up all utterences in the
    dataset:

    self.back_channels = ['right', 'okay', 'mmhmm', 'uh-huh', 'yeah', 'mm']

    This class stores all extracted backchannels in a list where each entry in
    the list is a dict:

        {'name': name, 'time', time, 'sample': sample, 'words': words}

    Here 'name' is the name of the session, 'time' and 'sample' are the timings of
    the utterence and 'words' is a list of strings.
    '''

    def __init__(self, pause=0.2, pre_padding=0, post_padding=0, max_utterences=1, root=None):
        self.paths = get_paths(root)
        self.session_names = self._session_names()

        self.pause = pause
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.max_utterences = max_utterences
        self.back_channels = ['right', 'okay', 'mmhmm', 'uh-huh', 'yeah', 'mm']

        f_data, g_data = self.extract_all_short_utterence_from_both_users()
        self.f_utter, self.f_vocab = self.get_utterences(f_data)
        self.g_utter, self.g_vocab = self.get_utterences(g_data)

        self.back_channel_list = self.get_back_channel_list()
        # self.add_audio_signal_to_back_channel_data()

    def _session_names(self):
        return [fname.split('.')[0] for fname in \
                os.listdir(self.paths['dialog_path']) if fname.endswith('.wav')]

    def extract_tag_data_from_xml_path(self, xml_path=None):
        '''
        Extract timed-unit (tu), silence (si) and noise (noi) tags from the
        .xml annotation file.
        '''

        if not xml_path:
            raise OSError("xml path required")

        # parse xml
        xml_element_tree = ET.parse(xml_path)

        tu, words, sil, noi = [], [], [], []
        for elem in xml_element_tree.iter():
            try:
                tmp = (float(elem.attrib['start']), float(elem.attrib['end']))
            except:
                continue
            if elem.tag == 'tu':
                # elem.attrib: start, end, utt
                words.append(elem.text)  # word annotation
                tu.append(tmp)
            elif elem.tag == 'sil':
                # elem.attrib: start, end
                sil.append(tmp)
            elif elem.tag == 'noi':
                # elem.attrib: start, end, type='outbreath/lipsmack/...'
                noi.append(tmp)

        return {'tu': tu, 'silence': sil, 'noise': noi, 'words': words}

    def merge_pauses(self, tu, words):
        new_tu, new_words = [], []
        start, last_end, tmp_words = 0, 0, []
        for i, (t, w) in enumerate(zip(tu, words)):
            # t[0] - start,  t[1] - end
            pause_duration = t[0] - last_end
            if pause_duration > self.pause:
                new_tu.append((start, last_end))
                new_words.append(tmp_words)
                tmp_words = [w]
                start = t[0]
                last_end = t[1]
            else:
                tmp_words.append(w)
                last_end = t[1]
        return new_tu[1:], new_words[1:]  # remove first entry which is always zero

    def get_timing_utterences(self, name, user='f'):

        # load timed-units.xml. Searching through dir.
        for file in os.listdir(self.paths['timed_units_path']):
            if name in file:
                if '.'+user+'.' in file:
                    xml_path = join(self.paths['timed_units_path'], file)

        data = self.extract_tag_data_from_xml_path(xml_path)

        times, words = self.merge_pauses(data['tu'], data['words'])

        samples = librosa_time_to_samples(times, sr=20000)

        return [{'name': name, 'user': user, 'time':time, 'sample': sample, 'words': word} \
                for time, sample, word in zip(times, samples, words)]

    def extract_all_short_utterence_from_both_users(self):
        f_data, g_data = [], []
        for name in tqdm(self.session_names):
            f_data.append(self.get_timing_utterences(name, user='f'))
            g_data.append(self.get_timing_utterences(name, user='g'))
        return f_data, g_data

    def get_utterences(self, data):
        utterence_data, vocab = [], {}
        for session in data:
            tmp_session_data = []
            for utterence in session:
                utter = utterence['words']
                if len(utter) <= self.max_utterences:
                    tmp_session_data.append(utterence)
                    if not utter[0] in vocab.keys():
                        vocab[utter[0]] = 1
                    else:
                        vocab[utter[0]] += 1
            utterence_data.append(tmp_session_data)

        vocab = sorted(vocab.items(), key=lambda t: t[1], reverse=True)
        return utterence_data, vocab

    def get_back_channel_list(self):
        back_channel_list = []
        for file_utters in self.f_utter + self.g_utter:
            for utter in file_utters:
                word = utter['words'][0]
                if word in self.back_channels:
                    back_channel_list.append(utter)
        return back_channel_list
```

## Maptask

This dataset goes through the annotations and extract relevant backchannels.

A pause of 0.1 seconds means that all utterences seperated by less than 0.1 seconds are
concatonated into longer utterences. Then 

```python
pause = 0.1
max_utterences = 1

maptask = Maptask(pause, max_utterences)

print(len(maptask.back_channel_list))
```


## Dataset

Pytorch Dataset

```python
class MaptaskDataset(Dataset):
    # 128 dialogs, 16-bit samples, 20 kHz sample rate, 2 channels per conversation
    def __init__(self,
                 pause=0.5,
                 pre_padding=0,
                 post_padding=0,
                 max_utterences=1,
                 sample_rate=20000,
                 window_size=50,
                 hop_length=None,
                 n_fft=None,
                 pad=0,
                 n_mels=40,
                 torch_load_audio=True,
                 audio=None,
                 normalize_audio=True,
                 root=None):
        self.maptask = Maptask(pause, pre_padding, post_padding, max_utterences, root)
        self.paths = self.maptask.paths

        # Audio
        self.normalize_audio = normalize_audio
        self.pre_padding = librosa_time_to_samples(pre_padding, sr=sample_rate)
        self.post_padding = librosa_time_to_samples(post_padding, sr=sample_rate)

        self.torch_load_audio = torch_load_audio
        if audio:
            self.audio = audio
        else:
            self.audio = self._audio()

        # Mel
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.pad = pad
        self.n_mels = n_mels

    def _audio(self):
        '''Iterate through the dialog diractory and extract all .wav files and
        store in dict'''
        audio = {}
        for wav in tqdm(os.listdir(self.paths['dialog_path'])):
            if '.wav' in wav:
                fpath = join(self.paths['dialog_path'], wav)
                name = wav.split('.')[0]  #  q1ec1.mix.wav -> q1ec1
                if self.torch_load_audio:
                    try:
                        if self.norm:
                            y, sr = torchaudio.load(fpath, normalization=self.norm)
                        else:
                            y, sr = torchaudio.load(fpath)
                        audio[name] = y
                    except:
                        # print(wav)
                        continue
                else:
                    try:
                        sr, y = read(fpath)
                        audio[name] = y
                    except:
                        # print(wav)
                        continue
        return audio

    def __len__(self):
        return len(self.maptask.back_channel_list)

    def __getitem__(self, idx):
        bc = self.maptask.back_channel_list[idx]
        start, end = bc['sample']

        start -= self.pre_padding
        end += self.post_padding

        # TODO
        # get correct audio tracks
        # Should also extract words of speaker not only backchannel
        y = self.audio[bc['name']]
        if bc['user'] == 'f':
            speaker = y[start:end, 0]
            back_channel = y[start:end,1]
        else:
            speaker = y[1, start:end]
            back_channel = y[0, start:end]
        return speaker, back_channel, bc['words']
```

Pytorch Dataset

```python
dset = MaptaskDataset(pause=0.1, max_utterences=1)

print('Amount of backchannels: ', len(dset))

speaker, bc, bc_word = dset[3]
print('speaker: ', len(speaker), type(speaker))
print('backchannel audio: ', len(bc), type(bc))
print('word: ', bc_word)
```
