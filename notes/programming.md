# Programming

## Utils

```python
from os.path import join
from tqdm import tqdm
import librosa
import numpy as np
import os
import xml.etree.ElementTree as ET
```

### Paths
Useful script to load a custom data path or generic which I try to use
across systems.

This one is helpful and could be booilerplate in all code.
A
function that could use a custom path or go to standard.

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

## Extract tag data from xml path

Given an xml-path extract annotated timed-
units data. The annotations includes time-units for words spoken, silence and
random noise. 

The data is returned as a dict containing the spoken words,
their timings, the times for noise and silence.

```python
def extract_tag_data_from_xml_path(xml_path):
    '''
    Extract timed-unit (tu), silence (si) and noise (noi) tags from the
    .xml annotation file.
    '''
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
```

```python
def get_timing_utterences(name,
                          user='f',
                          pause_time=1,
                          pre_padding=0,
                          post_padding=0,
                          timed_units_path=None):

    def merge_pauses(tu, words, threshold=0.1):
        new_tu, new_words = [], []
        start, last_end, tmp_words = 0, 0, []

        for i, (t, w) in enumerate(zip(tu, words)):
            # t[0] - start,  t[1] - end
            pause_duration = t[0] - last_end
            if pause_duration > threshold:
                new_tu.append((start, last_end))
                new_words.append(tmp_words)
                tmp_words = [w]
                start = t[0]
                last_end = t[1]
            else:
                tmp_words.append(w)
                last_end = t[1]
        return new_tu[1:], new_words[1:]  # remove first entry which is always zero

    if not timed_units_path:
        timed_units_path = get_paths()['timed_units_path']

    # load timed-units.xml. Searching through dir.
    for file in os.listdir(timed_units_path):
        if name in file:
            if '.'+user+'.' in file:
                xml_path = join(timed_units_path, file)

    data = extract_tag_data_from_xml_path(xml_path)
    times, words = merge_pauses(data['tu'], data['words'])

    # Pad utterence to include context
    if pre_padding or post_padding:
        t = np.array(times)
        t += (-pre_padding, post_padding)
        times = t

    samples = librosa.time_to_samples(times, sr=20000)

    return [{'time':time, 'sample': sample, 'words': word} \
            for time, sample, word in zip(times, samples, words)]
```

```python
def get_utterences(all_sessions_data, utterence_length=1):
    utterence_data, vocab = [], {}
    for session in all_sessions_data:
        tmp_session_data = []
        session_data = session['data']
        for utterence in session_data:
            utter = utterence['words']
            if len(utter) <= utterence_length:
                tmp_session_data.append(utterence)
                if not utter[0] in vocab.keys():
                    vocab[utter[0]] = 1
                else:
                    vocab[utter[0]] += 1
        utterence_data.append({'name': session['name'],
                                    'data': tmp_session_data})
    return utterence_data, vocab


```

## Find how many backchannels present in maptask

1. Get all session name in
maptask audio folder
2. Extract all timing and word data from each session
(entire dataset)
3. Get all one word utterences
4. Look at one word vocab and
extract backchannels

```python
# PATHS
paths = get_paths()

session_names = [fname.split('.')[0] for fname in \
                 os.listdir(paths['dialog_path']) if fname.endswith('.wav')]

print('dialog path: ', paths['dialog_path'])
print('annotation path: ', paths['annotation_path'])
print('mono path: ', paths['mono_path'])
```

### Extract all short utterences for the follower

```python
user = 'f'
all_f_data = []
for name in tqdm(session_names):
    session_data = get_timing_utterences(name,
                                         user=user,
                                         pause_time=0.2,
                                         pre_padding=0,
                                         post_padding=0,
                                         timed_units_path=paths['timed_units_path'])
    all_f_data.append({'name': name, 'data':session_data})
```

Extract all short utterences for the guide

```python
user = 'g'
all_g_data = []
for name in tqdm(session_names):
    session_data = get_timing_utterences(name,
                                         user=user,
                                         pause_time=0.2,
                                         pre_padding=0,
                                         post_padding=0,
                                         timed_units_path=paths['timed_units_path'])
    all_g_data.append({'name': name, 'data':session_data})
```

```python
f_small_utterences, f_vocab = get_utterences(all_f_data, utterence_length=1)
g_small_utterences, g_vocab = get_utterences(all_g_data, utterence_length=1)

f_vocab = sorted(f_vocab.items(), key=lambda t: t[1], reverse=True)
g_vocab = sorted(g_vocab.items(), key=lambda t: t[1], reverse=True)

print('Follower Vocab:')
for entry in f_vocab[:10]:
    print(entry)
```

```python
print('Guide Vocab:')
for entry in g_vocab[:10]:
    print(entry)
```

```python
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

print('Follower:')
print('Datapoints: ', f_dpoints)
print('Vocab: ', f_back)
```
