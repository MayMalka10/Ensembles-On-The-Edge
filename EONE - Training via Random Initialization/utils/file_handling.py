def strip_prefix(state_dict, prefix='encoder_decoder'):
    new_state_dict = {}
    for k in state_dict:
        if k.startswith(prefix):
            new_state_dict[k.split(prefix+'.')[1]] = state_dict[k]
    return new_state_dict