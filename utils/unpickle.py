import pickle

def read_pickle(path, name, extension):
    with open(f'{path}/{name}.{extension}', 'rb') as f:
        out = pickle.load(f)
    return out
