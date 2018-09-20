# Maptask

1. [download-audio.sh](download-audio.sh)
    - Downloads data (wavs and annotation) into the [data](data/) directory
2. [preprocess.py](preprocess.py)
    - Read annotations for maptask and extract what is defined as backchannel utterences.
    - Chop out the relevant parts of the audio and save as .npy files
3. [dataset.py](dataset.py)
  - `DSet`: A simple data set which uses preextracted pitch and intensity
4. [training.py](training.py)
  - Trains a basic lstm model to predict the pitch and intensity given previous context.

