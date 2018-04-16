def remove_module(state):
    new_state = {}
    for key, val in state.items():
        new_state[key[key.index(".") + 1:]] = val
    return new_state
