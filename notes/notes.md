# Notes for Maptask

[HCRC Map Task](http://journals.sagepub.com/doi/abs/10.1177/002383099103400404) corpus (Anderson et al., 1991) 

**Possible Features**
* Voice activity:
  * binary feature (speech/no speech)
  * backchannels
* Pitch:
  * Current pitch level
  * transformed into semitones and z-normalized
  * Relative and absolute values
* Power:
  * MelSpectrogram
  * Intensity (dB) extracted by Snack and z-normalized for each speaker
* Spectral stability:
  * Final lengthening is known to be an indicator for turn-taking
  * extract power spectrum of N band (up to 4khz) using Snack FFT
  * Calculate stability
        * S = sum(p) - sum( abs(p-p_prev) )
* Part-of-speech (POS)
  - Verbs, nouns, adj, etc


