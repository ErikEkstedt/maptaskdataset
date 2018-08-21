'''
Extracts mono-channels and GeMAP fetures and saves to disk.

This script does the audio processing which is saved on disk for the
maptask-dataset. The script asserts that both the dialogue wav-files and the
annotation folder is downloaded locally.

1.  Converts the dialogue dual-channel audio to two separate wav-files. One denoted
    'f' for follower and the other as 'g' for guide.

2.  Using OpenSmile GeMAP features are extracted for each of the mono-files and
    stores the features on disk as csv-files.
'''


# Assumes maptask dialogue files are downloaded in $dataPath
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
data_path = os.path.realpath(join(path, 'data'))
dialogue_path = join(data_path, 'dialogues')
mono_path = join(data_path, 'dialogues_mono')
gemap_path = join(data_path, 'gemaps')
opensmile_path = os.path.realpath(join(path, '..', '..', 'opensmile/opensmile-2.3.0'))


# 1. Seperate channels in audio.
convert = input('Convert stero to mono? (y/n)')
if convert == 'y' or convert == 'Y':
    from utils import convertStereoToMono
    convertStereoToMono(dialogue_path, mono_path, sr=20000)


# 2. Extract GeMAPs features from monofiles. Write to csv.
extract = input('Extract GeMAPS from monofiles with OpenSmile? (y/n)')
if extract == 'y' or extract == 'Y':
    from utils import extractGeMAPS
    extractGeMAPS(mono_path, gemap_path, opensmile_path)


# 3. Maptask -> Tacotron
extract = input('Reformat Maptask data -> Tacotron data')
if extract == 'y' or extract == 'Y':
    from utils import maptask_to_tacotron
    output_path = join(data_path, 'tacotron_style')
    timed_units_path = join(data_path, "maptaskv2-1/Data/timed-units")

    if not exists(output_path):
        print('Outputpath does not exist!')
        print('Creating  at:')
        print(output_path)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Reads
    maptask_to_tacotron(output_path,
                        timed_units_path,
                        mono_path,
                        pause_time=1,
                        sr=20000)
