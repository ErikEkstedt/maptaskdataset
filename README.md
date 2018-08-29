# Maptask

## Install

1. `conda create -n maptask python=3.6`
2. `source activate maptask`
3. `pip install -r requirements.txt`
4. `cd` into root of repo and `pip install -e .`

---------------------

* [download-audio.sh](./download-audio.sh)
  - downloads data (wavs and annotation) into the [data](./data/) directory
* [chop-audiofiles.sh](./chop-audiofiles.sh) 
  - Chops audio files in $1 into $3 second clips and stores in directory $2 ($x=input args when running script)
* [utils.py](./utils.py)
  - Contain the code for the various processing.
* [process\_audio.py](./process_audio.py)
  - Code for actually utilizing the code in `utils.py`
* [maptaskdata.py](./maptaskdata.py)


## Notebook

[notebook status](https://github.com/ErikEkstedt/maptaskdataset/blob/master/maptask/notes/programming.ipynb)


## Step 1

The goal of this code is to extract backchannel data from the maptask dataset in order
to train a model to generate backchannels conditioned on audio input. A speech-to-speech
model that hopefully captures some prosodic nuance and timings for backchannel generation.

1. Merge together utterences defined in one speaker that are seperated less than `$pause`
2. Get all utterences consisting of only one word. Decide on which one word phrases are
   constituting a backchannel.
3. Define how large the `context` is, i.e how many seconds before the backchannel is part of
   the datapoint.
4. Extract all backchannels with `context` as audio and text
5. [x] Create PyTorch Dataset and DataLoader


