# Maptask


* [download-audio.sh](./download-audio.sh)
  - downloads data (wavs and annotation) into the [data](./data/) directory
* [chop-audiofiles.sh](./chop-audiofiles.sh) 
  - Chops audio files in $1 into $3 second clips and stores in directory $2 ($x=input args when running script)
* [utils.py](./utils.py)
  - Contain the code for the various processing.
* [process\_audio.py](./process_audio.py)
  - Code for actually utilizing the code in `utils.py`



## Install

1. `conda create -n maptask python=3.6`
2. `source activate maptask`
3. `pip install -r requirements.txt`

## Programming utils

[notebook status](/notes/programming.ipynb)

