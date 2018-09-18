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
# import torchaudio
from torch.utils.data import Dataset, DataLoader


def current_path(root=None, verbose=False):
    '''
    Arguments:
            root:   path to maptask data download.
                    contains dirs: dialogues, maptaskv2-1
            verbose: boolean, to print out paths
    '''
    if not root:
        full_path = os.path.realpath(__file__)
        root_path, filename = os.path.split(full_path)
    else:
        root_path = root
    data_path = os.path.join(root_path, 'data')
    if verbose:
        print('full_path', full_path)
        print('root_path', root_path)
        print('data_path', data_path)
        print('filename', filename)
    # root_path: GitRepositoryRoot/maptask
    return {'root' : root_path,
            'data' : data_path,
            'annotations' : join(data_path, 'maptaskv2-1'),
            'dialogues' : join(data_path, 'dialogues'),
            'savepath' : join(data_path, 'processed'),
            'timed_units_path': join(data_path, "maptaskv2-1/Data/timed-units"),
            'gemap_path' : join(data_path, 'gemaps'),
            'opensmile_path' : join(os.path.expanduser('~'), 'opensmile-2.3.0')}


# Annotation
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

    def __init__(self, pause=0.5, max_utterences=1, root=None, n_files=None):
        self.paths = current_path(root)
        self.n_files = n_files
        self.session_names = self._session_names()

        self.pause = pause
        self.max_utterences = max_utterences
        self.back_channels = ['right', 'okay', 'mmhmm', 'uh-huh', 'yeah', 'mm']

        f_data, g_data = self.extract_all_short_utterence_from_both_users()
        self.f_utter, self.f_vocab = self.get_utterences(f_data)
        self.g_utter, self.g_vocab = self.get_utterences(g_data)

        self.back_channel_list = self.get_back_channel_list()

    def _session_names(self):
        session_names = [fname.split('.')[0] for fname in \
                         os.listdir(self.paths['dialogues']) if fname.endswith('.wav')]
        if self.n_files:
            print('n_files', self.n_files)
            session_names = session_names[:self.n_files]
        return session_names

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


# Preprocess
def save_bc(y, bc, filename, context=2, sr=20000):
    start, end = bc['sample']

    n_samples = librosa_time_to_samples(context, sr=sr)
    context = np.zeros(n_samples)
    back_channel = np.zeros(n_samples)

    # Find start of context >= 0
    context_start = end - n_samples
    if context_start < 0:
        context_start = 0

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
    back_channel_class = np.zeros(back_channel.shape)
    back_channel_class[-n_samples_bc:] = 1

    data = np.array((context, back_channel, back_channel_class))
    np.save(filename, data)


def chunk_audio(bc_list, loadpath, savepath, context=2, sr=20000):
    ''' Chunk and store audio as .np
    Arguments:
        bc_list: list of backchannel infor
        loadpath: path to load wav files from
        savepath: path to store .np
    '''
    pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
    name = ''
    n = 0
    for bc in tqdm(bc_list):
        if name != bc['name']:
            name = bc['name']
            fpath = join(loadpath, name + '.mix.wav')
            # TODO
            # Only thing I ended up using torchaudio for?
            # y, sr = torchaudio.load(fpath)
            # y = y.numpy()
            sr, y = read(fpath)
            y = y.as_type(np.float)
            filename = join(savepath, name + '_' + str(n))
            save_bc(y, bc, filename, context, sr)
        else:
            filename = join(savepath, name + '_' + str(n))
            save_bc(y, bc, filename, context, sr)
        n += 1


def process_audio(pause=0.5, max_utterences=1, context=2, savepath='data/processed'):
    '''
    Read annotations for maptask and extract what is defined as backchannel
    utterences.

    Chop out the relevant parts of the audio and save as .np files
    '''
    print('Reading wav files:')
    print('Pause: {}, Max utterences: {}'.format(pause, max_utterences))
    maptask = Maptask(pause=pause, max_utterences=max_utterences)
    print('-'*55)
    print('Removing corrupt file q6ec2.mix.wav')
    try:
        corrupt = os.path.join(maptask.paths['dialogues'], 'q6ec2.mix.wav')
        os.remove(corrupt)
    except:
        pass
    print('-'*55)
    print('Total backchannels found: ', len(maptask.back_channel_list))
    print('Cut out backchannels and store as .np')
    print('Store in: ', savepath)
    chunk_audio(maptask.back_channel_list,
                maptask.paths['dialogues'],
                savepath=maptask.paths['savepath'],
                context=2)


if __name__ == "__main__":
    print('Preprocessing using defaults')
    process_audio()
