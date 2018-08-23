'''
Extracts mono-channels and GeMAP fetures and saves to disk.

This script does the audio processing which is saved on disk for the
maptask-dataset. The script asserts that both the dialogue wav-files and the
annotation folder is downloaded locally.

1.  Converts the dialogue dual-channel audio to two separate wav-files. One denoted
    'f' for follower and the other as 'g' for guide.

2.  Using OpenSmile GeMAP features are extracted for each of the mono-files and
    stores the features on disk as csv-files.

3.  Maptask -> Tacotron. Tacotron utilizes short audio clips with correlating
    word annotations. In NVIDIAS implementation the data is stored in short
    wav-files and a csv where each row includes the filename and the associated
    utterence string.
'''

import os
from os.path import join, exists
import pathlib
from utils import get_paths

paths = get_paths()

# 1. Seperate channels in audio.
convert = input('Convert stero to mono? (y/n) ')
if convert == 'y' or convert == 'Y':
    from utils import convertStereoToMono
    convertStereoToMono(paths['dialogue_path'], paths['mono_path'], sr=20000)


# 2. Extract GeMAPs features from monofiles. Write to csv.
extract = input('Extract GeMAPS from monofiles with OpenSmile? (y/n) ')
if extract == 'y' or extract == 'Y':
    from utils import extractGeMAPS
    extractGeMAPS(paths['mono_path'], paths['gemap_path'], paths['opensmile_path'])


# 3. Maptask -> Tacotron
extract = input('Reformat Maptask data -> Tacotron data (y/n) ')
if extract == 'y' or extract == 'Y':
    from utils import maptask_to_tacotron

    output_path = join(paths['data_path'], 'tacotron_style')

    if not exists(output_path):
        print('Outputpath does not exist!')
        print('Creating  at:')
        print(output_path)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Reads
    maptask_to_tacotron(output_path,
                        paths['timed_units_path'],
                        paths['mono_path'],
                        pause_time=1,
                        sr=20000)
