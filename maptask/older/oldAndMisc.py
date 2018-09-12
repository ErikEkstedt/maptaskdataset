# Maptask __getitem__() where context as "data" and back_channel as "target"

def get_segments(self, idx):
    bc = self.maptask.back_channel_list[idx]
    start, end = bc['sample']

    # transform time-padding -> sample-padding and add to start, end
    context_start = start - self.context
    context_end = start

    # TODO: Should also extract words of speaker not only backchannel
    y = self.audio[bc['name']]  # load correct audio array
    if bc['user'] == 'f':
        # back channel generator is 'f'
        context = y[context_start:context_end, 0]
        self_context = y[context_start:context_end, 1]

        back_channel = y[start:end,1]
    else:
        # back channel generator is 'g'
        context = y[context_start:context_end, 1]
        self_context = y[context_start:context_end, 0]

        back_channel = y[start:end,0]

    # Context
    context_spec = self.mel_spec(context.unsqueeze(0))
    self_context_spec = self.mel_spec(self_context.unsqueeze(0))

    # Target - backchannel
    bc_spec = self.mel_spec(back_channel.unsqueeze(0))

    return {'context_audio': context,
            'context_spec': context_spec,
            'self_context_audio': self_context,
            'self_context_spec': self_context_spec,
            'back_channel_audio': back_channel,
            'back_channel_spec': bc_spec,
            'back_channel_word': bc['words'][0]}

def sound_datapoint(output, with_self_context=False, sr=20000):
    sd.default.samplerate = sr
    context = output['context_audio'].numpy()
    bc = output['back_channel_audio'].numpy()
    bc_word = output['back_channel_word']
    print('context audio: ', context.shape)
    if with_self_context:
        self_context = output['self_context_audio'].numpy()
        audio = np.vstack((self_context, context)).T
        sd.play(audio)
        time.sleep(librosa.get_duration(context, sr=20000))
    else:
        sd.play(context)
        time.sleep(librosa.get_duration(context, sr=20000))
    print('BC audio: ', bc_word)
    sd.play(bc)
    time.sleep(librosa.get_duration(bc, sr=20000))
