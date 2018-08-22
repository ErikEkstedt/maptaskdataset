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

def convertStereoToMono(data_path, outputPath, sr=20000):
    if not exists(data_path):
        print('datapath does not exist!')
        print('nonexistent -> ', data_path)
        print('Please download files by running audio/download-audio.sh')
        exit()

    if not exists(outputPath):
        print('Outputpath does not exist!')
        print('Creating  at:')
        print(outputPath)
        pathlib.Path(outputPath).mkdir(parents=True, exist_ok=True)

    print('-'*50)
    print('Reading stereo .wav-files in:')
    print(data_path)
    print('-'*50)
    print('Saving as mono .wav-files in:')
    print(outputPath)
    print()

    wavFilenames = os.listdir(data_path)
    wavFilenames.sort()  # Nice to do in order when debugging
    for wavFile in tqdm(wavFilenames):
        if '.wav' in wavFile:
            fpath = join(data_path, wavFile)  # path to file
            sessionName = wavFile.split('.')[0]  # example: q1ec1 (removed .mix.wav)

            try:
                # (g, f), _ = librosa.load(fpath, sr=sr, mono=False)  # y(t), Extract audio array
                r, arr = read(fpath)
            except:
                print('Could not read ', wavFile)
                continue

            g, f = np.hsplit(arr, 2)

            write(filename=join(outputPath, sessionName+'.f.wav'),
                  rate=r,
                  data=f)

            write(filename=join(outputPath, sessionName+'.g.wav'),
                  rate=r,
                  data=g)

    print('Done! Check files in \n', outputPath)
    print('-'*50)


def extractGeMAPS(data_path, output_path, opensmile_path):
    '''
    Processes wav-file and extracts gemaps by calling OpenSmile.
    The gemaps are saved on disk as csv files.

    Input
    :inFilePath         Path to input file (.wav)
    :output_path         Path to output folder where csv is stored

    Return
    :outFilePath        Path to specific csv file
    '''

    if not os.path.exists(opensmile_path):
        print('\nPath to OpenSmile directory does not exist!')
        print(opensmile_path)
        print('GeMAP feature extraction not possible...')
        print('Exiting')
        return 0

    # OpenSMILE paths
    uname = os.uname()[0]
    if uname in "Darwin":
        SMILEExtract = join(opensmile_path, "inst/bin/SMILExtract")
    else:  # Assume linux
        SMILEExtract = join(opensmile_path, "bin/linux_x64_standalone_libstdc6/SMILExtract")

    confPath = join(opensmile_path, "config")
    gemapsConf = join(confPath, "gemaps/eGeMAPSv01a.conf")

    if not os.path.exists(output_path):
        print('\nTarget path does not exist!')
        print('Creating  path at:')
        print(output_path)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    counter = 0
    wavFilenames = os.listdir(data_path)
    wavFilenames.sort()  # Nice to do in order when debugging
    for wavFile in tqdm(wavFilenames):
        if '.wav' in wavFile:
            fpath = join(data_path, wavFile) # Full path to file
            outFilePath = join(output_path, wavFile[:-3] + 'csv')  # wav -> csv

            c=[SMILEExtract,
            '-C', gemapsConf,     # config filepath
            '-I', fpath,          # Input filepath
            '-instname', wavFile, # -instname: Usually the input filename, saved in first column in CSV and ARFF output.
            '-D', outFilePath,    # Destination filepath
            '-loglevel', '0',     # -l, -loglevel: Verbosity level. 0=show nothing, gradually more verbose
            '-timestampcsv 1']
            c = " ".join(c)

            # Call opensmile, time it and some error handling
            returnCode = call(c, shell=True)
            if returnCode > 0:
                print('File: ', )
                print('Exit code: ', returnCode)
            else:
                counter += 1
    print('Extracted GeMAP features from {} files.'.format(counter))


def extract_tag_data(name, annotation_path):
    '''
    Extract timed-unit (tu), silence (si) and noise (noi) tags from the
    .xml annotation file.

    An xml-file represents a recorded dialog, e.g q1ec1.wav.
    '''
    # load timed-units.xml
    timed_units_path = join(annotation_path, 'Data/timed-units')
    tu_path = join(timed_units_path, name + '.timed-units.xml')
    elements = ET.parse(tu_path)

    tu, words, sil, noi = [], [], [], []
    for elem in elements.iter():
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


# ------------------------------------------

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



    def create_data_points(session_name, annotation_path, user='f'):
        print(session_name)
        # load timed-units.xml
        timed_units_path = join(annotation_path, 'Data/timed-units')
        for f in os.listdir(timed_units_path):
            if session_name in f and '.'+user+'.' in f:
                print(f)
                user_data = get_time_filename_utterence(f, annotation_path, pause_time=0.2)



# ------------------------------------------

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


def get_timing_utterences(name,
                          user='f',
                          pause_time=1,
                          pre_padding=0,
                          post_padding=0,
                          annotation_path=None):

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

    if not annotation_path:
        annotation_path = get_paths()['annotation_path']

    # load timed-units.xml. Searching through dir.
    timed_units_path = join(annotation_path, 'Data/timed-units')
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


def load_dialog(name, dialog_path=None):
    if not dialog_path:
        dialog_path = get_paths()['dialog_path']
    for file in os.listdir(dialog_path):
        if name in file:
            if file.endswith('.wav'):
                return read(join(dialog_path, file))



def get_data_from_all(session_names):
    all_sessions_data = []
    for name in tqdm(session_names):
        session_data = get_timing_utterences(name,
                                    user='f',
                                    pause_time=0.2,
                                    pre_padding=0,
                                    post_padding=0)
        all_sessions_data.append({'name': name, 'data':session_data})


def get_backchannel_data(all_sessions_data, utterence_length=1):
    back_channel_data, vocab = [], {}
    for session in all_sessions_data:
        tmp_session_data = []
        session_data = session['data']
        for utterence in session_data:
            utter = utterence['words']
            if len(utter) <= utterence_length:
                tmp_session_data.append(utterence)
                if not utter[0] in counter.keys():
                    counter[utter[0]] = 1
                else:
                    counter[utter[0]] += 1
        back_channel_data.append({'name': session['name'],
                                    'data': tmp_session_data})
    return back_channel_data, counter


def listen_to_data(name, session_data):
    for d in session_data:
        print(d['words'])
        t0, t1 = d['time']
        s0, s1 = d['sample']
        print('time: {} - {}'.format(t0, t1))
        print('sample: {} - {}'.format(s0, s1))
        sr, y = load_dialog(name)
        sd.play(y[s0:s1])
        t0, t1 = d['time']
        duration = t1 - t0
        time.sleep(duration)



def test():
    import matplotlib.pyplot as plt
    import sounddevice as sd
    import random
    import time

    sd.default.samplerate = 20000

    # get_paths()
    paths = get_paths()
    dialog_path = paths['dialog_path']
    annotation_path = paths['annotation_path']
    mono_path = paths['mono_path']
    print('dialog path: ', dialog_path)
    print('annotation path: ', annotation_path)
    print('mono path: ', mono_path)


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

if __name__ == "__main__":
    test()
