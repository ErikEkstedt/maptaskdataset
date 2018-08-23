import os
from tqdm import tqdm
from os.path import join
import xml.etree.ElementTree as ET
from librosa import time_to_samples as librosa_time_to_samples

from utils import get_paths


class Maptask(object):

    def __init__(self, pause=0.2, pre_padding=0, post_padding=0, max_utterences=1, root=None):
        self.paths = get_paths(root)
        self.session_names = self._session_names()

        self.pause = pause
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.max_utterences = max_utterences

        f_data, g_data = self.extract_all_short_utterence_from_both_users()
        self.f_utter, self.f_vocab = self.get_utterences(f_data)
        self.g_utter, self.g_vocab = self.get_utterences(g_data)

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

        return [{'time':time, 'sample': sample, 'words': word} \
                for time, sample, word in zip(times, samples, words)]

    def extract_all_short_utterence_from_both_users(self):
        f_data, g_data = [], []
        for name in tqdm(self.session_names):
            f_data.append({'name': name, 'data': self.get_timing_utterences(name, user='f')})
            g_data.append({'name': name, 'data': self.get_timing_utterences(name, user='g')})
        return f_data, g_data

    def get_utterences(self, data):
        utterence_data, vocab = [], {}
        for session in data:
            tmp_session_data = []
            session_data = session['data']
            for utterence in session_data:
                utter = utterence['words']
                if len(utter) <= self.max_utterences:
                    tmp_session_data.append(utterence)
                    if not utter[0] in vocab.keys():
                        vocab[utter[0]] = 1
                    else:
                        vocab[utter[0]] += 1
            utterence_data.append({'name': session['name'],
                                        'data': tmp_session_data})

        vocab = sorted(vocab.items(), key=lambda t: t[1], reverse=True)
        return utterence_data, vocab



if __name__ == "__main__":
    maptask = Maptask(pause=0.1)

    f_utter = maptask.f_utter
    f_vocab = maptask.f_vocab
    g_utter = maptask.g_utter
    g_vocab = maptask.g_vocab

    f_dpoints, f_back = 0, []
    for i in range(5):
        f_back.append(f_vocab[i][0])
        f_dpoints += f_vocab[i][1]

    g_dpoints, g_back = 0, []
    for i in range(5):
        g_back.append(g_vocab[i][0])
        g_dpoints += g_vocab[i][1]

    print('Guide:')
    print('Datapoints: ', g_dpoints)
    print('Vocab: ', g_back)
    print()
    print('Follower:')
    print('Datapoints: ', f_dpoints)
    print('Vocab: ', f_back)

    print('-'*50)






