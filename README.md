# Maptask

## Install

1. `conda create -n maptask python=3.6`
2. `source activate maptask`
3. `pip install -r requirements.txt`

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

[notebook status](/notes/programming.ipynb)


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


## Step 2

Define a dataset and dataloader to be used during training.

* What is a datapoint?
* Why do we construct the datapoint as such?

Thoughts
* Cleaning the audio?
  - All parts where silence is indicated should be set to zero.
  - Define a backchannel with a sufficient amount of silence preceding it. Make sure that
	it is a response utternce as opposed to a continuation
* Should the training be continuous on a frame to frame basis?
  - There is no specific backchannel classification but all audio produced by network is
	considered a backchannel. The argumentation being that the dataset has been structured
	in such a way as to only contain backchannel response.
  - Given any audio in produce audio out which has highest probability. Autoregression.
* Should the training be continuous on a segment basis?
  - this would make it such that a datpoint would be defined as some context, i.e time
	with the speaker audio/information, and then the correct answer, i.e the backchannel
	audio.
  - For this approach all backchannels are found, the context extracted and used as input
	and the backchannel as output.
  - This would also require negative datapoints to distinguish between context where no
	backchannel is provided and when they are provided.
