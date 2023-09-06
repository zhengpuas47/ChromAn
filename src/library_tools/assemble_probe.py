
def shuffle_readout(seq, readout_inds=[2,3,4], linker='t'):
    import random
    _segments = seq.split(linker)
    _readouts = [_s for _i, _s in enumerate(_segments) if _i in readout_inds]
    random.shuffle(_readouts, )
    _reassemble_segments = []
    for _i, _s in enumerate(_segments):
        if _i not in readout_inds:
            _reassemble_segments.append(_s)
        else:
            #print(list(readout_inds).index(_i))
            _reassemble_segments.append(_readouts[list(readout_inds).index(_i)])
    _reassemble_seq = linker.join(_reassemble_segments)
    return _reassemble_seq