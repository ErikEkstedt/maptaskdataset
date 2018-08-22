# Programming

## Utils

<strong>TODO: These are all inputs from utils some are note needed anymore </strong>

```python
from os.path import join, exists
from scipy.io.wavfile import read, write
from subprocess import call
from tqdm import tqdm
import librosa
import numpy as np
import os
import glob
import pathlib
import xml.etree.ElementTree as ET
```

Import dependencies and set sounddevice sample rate to 20000Hz

```python
import matplotlib.pyplot as plt
import sounddevice as sd
import random
import time

sd.default.samplerate = 20000
```

### Paths
Useful script to load a custom data path or generic which I try to use across systems.

This one is helpful and could be booilerplate in all code.
A function that could use a custom path or go to standard.

```python
def get_paths():
    # Assumes maptask dialogue files are downloaded in $dataPath
    try:
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
    except: # for ipython repl error
        path = os.path.realpath(os.getcwd())
    data_path = os.path.realpath(join(path, 'data'))
    timed_units_path = join(data_path, "maptaskv2-1/Data/timed-units")
    return {'data_path' : data_path,
            'annotation_path' : join(data_path, 'maptaskv2-1'),
            'dialog_path' : join(data_path, 'dialogues'),
            'mono_path' : join(data_path, 'dialogues_mono'),
            'gemap_path' : join(data_path, 'gemaps'),
            'opensmile_path' : os.path.realpath(join(path, '..', '..', 'opensmile/opensmile-2.3.0'))}
```

Load the paths to the data


```python
paths = get_paths()
dialog_path = paths['dialog_path']
annotation_path = paths['annotation_path']
mono_path = paths['mono_path']
print('dialog path: ', dialog_path)
print('annotation path: ', annotation_path)
print('mono path: ', mono_path)
```


## Get time stamp and utterences

////

--------------

\[[doc string in function should become this intro text](todoipynb)
Make a programming with gitsubmodules. like vim programming-eco-system. \]

--------------


```python
def get_time_filename_utterence(name, annotation_path, pause_time=1):
    '''
    Arguments
    ---------

    :name                   session name (string)
    :timed_units_path       path to timed-units annotaions (string)
    :pause_time             pause length in sexonds to define discrete utterences (int)

    Return
    ------

    :tu_data                (list) of dicts containing time, name and utterence.

    a dict contains:
        dict['time'] = (start,end)
        dict['name'] = q1ec1-0001
        dict['utterence'] = 'hello my name is'
    '''
    tu_data = []
    # name = name[:-4]  # remove .wav

    tmp_dict = extract_tag_data(name, annotation_path)

    # Extract time and word annotations for utterences seperated by at least
    # :pause_time (seconds)
    i, start, end = 0, 0, 0
    tmp_utterence = ''
    for j, (t, w) in enumerate(zip(tmp_dict['tu'], tmp_dict['words'])):
        pause = t[0] - end  # pause since last word
        if pause < pause_time:
            if j == 0:
                tmp_utterence += w
            else:
                tmp_utterence += ' ' + w
                end = t[1]
        else:
            utterence = tmp_utterence
            time = (start, end)
            tmp_name = name + '-{0:04}'.format(i)
            tu_data.append({'time':time, 'name':tmp_name, 'words': utterence})
            i += 1
            start, end = t
            tmp_utterence = w
    return tu_data
```


## create data points 

Input the session name you need time data from, the annotation\_path and a user
(optional), either 'f' for follower or 'g' for guide

```python
def create_data_points(session_name, annotation_path, user='f'):
  print(session_name)
  # load timed-units.xml
  timed_units_path = join(annotation_path, 'Data/timed-units')
  for f in os.listdir(timed_units_path):
	if session_name in f and '.'+user+'.' in f:
  print(f)
  user_data = get_time_filename_utterence(f, annotation_path, pause_time=0.2)
```

## Current main (test)

1. Get all session name in maptask audio folder
2. Extract all timing and word data from each session (entire dataset)
3. 

```python
# 1.
session_names = [fname.split('.')[0] for fname in os.listdir(dialog_path) \
				  if fname.endswith( '.wav' )]

# 2.
all_sessions_data = get_data_from_all(session_names) 
# 3. back_channels and vocab. Vocab to see what kind of utterences we have.
# This should be notebook
back_channel_data, vocab = get_backchannel_data(all_sessions_data, 1)

for sess in back_channel_data:
	listen_to_data(sess['name'], sess['data'])

vocab = sorted(vocab.items(), key=lambda t: t[1], reverse=True)
vocab[:20]
```

-----------------------------
# Older

## Maptask to Tacotron

Given up idea of tacotron but rewrote code to store data as stored for tacotron.

Not pursuing anymore for now.

```python
def maptask_to_tacotron(output_path,
                        timed_units_path,
                        mono_path,
                        pause_time=1,
                        sr=20000):
    '''
    This script should extract the text and audio part of utterences seperated
    by $pause_time from $person in the Maptask dataset.

    The audio snippets is cut out and stored as wav-files with a name
    according to (ex: q1ec1-0001, q1ec1-0002, ...).

    Each line in the produces txt-file has the following form:

    name_of_audio|utterence (string)
    name_of_audio|utterence (string)
    '''
    mono_file_names = os.listdir(mono_path)
    mono_file_names.sort()  # Nice to do in order when debugging

    file_txt = join(output_path, 'maptask')
    file_f = open(file_txt+'.f.txt', "w")
    file_g = open(file_txt+'.g.txt', "w")

    wavs_path = join(output_path, 'wavs')
    if not exists(wavs_path):
        pathlib.Path(wavs_path).mkdir(parents=True, exist_ok=True)

    # Iterate through all (mono) audiofiles, chop the audio in to utterences
    for mono_wav in tqdm(mono_file_names):
        if '.wav' in mono_wav:  # failsafe
            # mono_wav: q1ec1.f.wav, q1ec1.g.wav, ...
            fpath = join(mono_path, mono_wav) # Full path to file

            # Load audio file
            sr, y = read(fpath)

            # get time and words from timed-units
            tu_data = get_time_filename_utterence(mono_wav, timed_units_path)

            for d in tu_data:
                start, end = d['time']  # time
                start = librosa.time_to_samples(start, sr=sr)
                end = librosa.time_to_samples(end, sr=sr)
                y_tmp = y[start:end]

                # write chopped audio to disk
                tmp_path = join(wavs_path, 'wavs', d['name']+'.wav')
                write(filename=tmp_path, rate=sr, data=y_tmp)

                # write corresponding row in txt
                s = d['name'] + '|' + d['words'] + '\n'
                if '.f.' in mono_wav:
                    file_f.write(s)
                else:
                    file_g.write(s)
    file_f.close()
    file_g.close()
```




