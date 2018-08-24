import os
from tqdm import tqdm
from scipy.io.wavfile import read, write
from os.path import join
import xml.etree.ElementTree as ET
from librosa import time_to_samples as librosa_time_to_samples

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MEL2 as MelSpectrogram
from torchaudio.transforms import SPECTROGRAM as Spec

from utils import get_paths

class Maptask(object):

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

        # Pad utterence to include context
        if self.pre_padding or self.post_padding:
            t = np.array(times)
            t += (-pre_padding, post_padding)
            times = t

        samples = librosa_time_to_samples(times, sr=20000)

        return [{'name': name, 'user': user, 'time':time, 'sample': sample, 'words': word} \
                for time, sample, word in zip(times, samples, words)]

    def extract_all_short_utterence_from_both_users(self):
        f_data, g_data = [], []
        for name in tqdm(self.session_names):
            # f_data.append({'name': name, 'data': self.get_timing_utterences(name, user='f')})
            # g_data.append({'name': name, 'data': self.get_timing_utterences(name, user='g')})
            f_data.append(self.get_timing_utterences(name, user='f'))
            g_data.append(self.get_timing_utterences(name, user='g'))
        return f_data, g_data

    def get_utterences(self, data):
        utterence_data, vocab = [], {}
        for session in data:
            tmp_session_data = []
            # for utterence in session['data']:
            for utterence in session:
                utter = utterence['words']
                if len(utter) <= self.max_utterences:
                    tmp_session_data.append(utterence)
                    if not utter[0] in vocab.keys():
                        vocab[utter[0]] = 1
                    else:
                        vocab[utter[0]] += 1
            utterence_data.append(tmp_session_data)

            # name is attach in data points instead
            # utterence_data.append({'name': session['name'],
            #                             'data': tmp_session_data})

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

    def add_audio_signal_to_back_channel_data(self):
        for wav in tqdm(os.listdir(self.paths['dialog_path'])):
            if '.wav' in wav:
                # y, sr = torchaudio.load(join(maptask.paths['dialog_path'], wav))
                try:
                    sr, y = read(join(self.paths['dialog_path'], wav))
                except:
                    print(wav)
                    continue
                bcs = [f for f in self.back_channel_list if f['name'] in wav]
                for bc in bcs:
                    start, end = bc['sample']
                    bc['audio'] = y[start:end]

    def print_vocab(self, top=5):
        f_dpoints, f_back = 0, []
        g_dpoints, g_back = 0, []
        for i in range(top):
            f_back.append(self.f_vocab[i][0])
            f_dpoints += self.f_vocab[i][1]
            g_back.append(self.g_vocab[i][0])
            g_dpoints += self.g_vocab[i][1]
        print('Guide:')
        print('Datapoints: ', g_dpoints)
        print('Vocab: ', g_back)
        print()
        print('Follower:')
        print('Datapoints: ', f_dpoints)
        print('Vocab: ', f_back)
        print()
        print('Total: ', g_dpoints + f_dpoints)
        print('-'*50)



def listen_to_backchannels(back_channels_list, reverse=False):
    # for y in reversed(back_channels_list):
    def _play(y):
        print(y)
        print(y['words'])
        print(y['name'])
        print(y['user'])
        audio = y['audio']
        print(type(audio))
        print(audio.dtype)
        user = 1 if y['user'] == 'f' else 0
        sd.play(audio[:, user])
        time.sleep(0.5)
        sd.stop()
    if reverse:
        for y in back_channels_list:
            _play(y)
    else:
        for y in back_channels_list:
            _play(y)


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
                 root=None):
        self.maptask = Maptask(pause, pre_padding, post_padding, max_utterences, root)
        self.paths = self.maptask.paths
        self.audio = self._audio()
        self.pre_padding = pre_padding
        self.post_padding = post_padding

        # Mel
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.pad = pad
        self.n_mels = n_mels

    def _audio(self):
        audio = {}
        for wav in tqdm(os.listdir(self.paths['dialog_path'])):
            if '.wav' in wav:
                try:
                    sr, y = read(join(self.paths['dialog_path'], wav))
                    name = wav.split('.')[0]
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

        # TODO
        # Make audio torch.Tensor (quantify)
        # Spectrogram
        # mel_spec = MelSpectrogram(sr=self.sample_rate,
        #                           ws=self.window_size,
        #                           hop=self.hop_length,
        #                           n_fft=self.n_fft,
        #                           pad=self.pad,
        #                           n_mels=self.n_mels)

        # s = mel_spec(torch.tensor(speaker)
        # print(speaker.shape)
        # print(back_channel.shape)
        # print(s.shape)

        # print(mel_spec.shape)
        return speaker, back_channel, bc['words']


if __name__ == "__main__":

    dset = MaptaskDataset()
    print(len(dset))
    speaker, bc, bc_word = dset[3]

    print(speaker)
    print(bc)
    print(bc_word)
    print(type(speaker))
    print(type(bc))
    print(type(bc_word))

    # dloader = DataLoader(dset, num_workers=4, batch_size=64)
    # for speaker, bc, bc_word in dloader:
    #     print(speaker.shape)
    #     print(bc.shape)

    # Spectrogram = Spec()
    # S = Spectrogram(g)
    # print('S.shape', S.shape)
    # print('S.shape', S.shape)
