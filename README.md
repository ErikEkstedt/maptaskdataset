# Maptask

## Install

`git clone https://github.com/ErikEkstedt/maptaskdataset.git`

1. `conda create -n maptask python=3.7`
    - [miniconda](https://conda.io/miniconda.html)
2. `source activate maptask`
3. `pip install -r requirements.txt`
4. `cd` into root of repo and `pip install -e .`
5. Install [torchaudio](https://github.com/pytorch/audio)
    - git clone [torchaudio](https://github.com/pytorch/audio)
    - Summary of installation
      - Dependencies:
        - Linux(Ubuntu): `sudo apt-get install sox libsox-dev libsox-fmt-all`
        - OSX: `brew install sox`
      - `cd` into root of repo and:
        - Linux: `python setup.py install`
        - OSX: `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install`


---------------------

## Preprocess

1. [download-audio.sh](download-audio.sh)
    - Downloads data (wavs and annotation) into the [data](data/) directory
2. [preprocess.py](preprocess.py)
    - Read annotations for maptask and extract what is defined as backchannel utterences.
    - Chop out the relevant parts of the audio and save as .npy files
3. [dataset.py](dataset.py)
    - `TransformDataset(Dataset)`
    - `class TranformDataloader(DataLoader)`
    - `plot_sample_example(dset, datatype='context', idx=None)`


## Notebooks

* [transforms](notes/transforms.ipynb)
  - Which packages to use in python for audio transformations
* [programming](notes/programming.ipynb)

## Knowledge Sources


* [Signal Processing](https://github.com/calebmadrigal/FourierTalkOSCON)
* [nnmnkwii_gallery](https://github.com/r9y9/nnmnkwii_gallery.git)
* [nnmnkwii](https://github.com/r9y9/nnmnkwii.git)
* [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
* [pysptk](https://github.com/r9y9/pysptk)
* [Tacotron](https://github.com/r9y9/pysptk)

([Cool Notebook source](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks#machine-learning-statistics-and-probability))

## Goal

The goal of this code is to extract backchannel data from the maptask dataset in order
to train a model to generate backchannels conditioned on audio input.

1. Merge together utterences defined in one speaker that are seperated less than `$pause`
2. Get all utterences consisting of only one word. Decide on which one word phrases are
   constituting a backchannel.
3. Define how large the `context` is, i.e how many seconds before the backchannel is part of
   the datapoint.
4. Extract all `backchannels` with `context`
5. Create PyTorch Dataset and DataLoader


