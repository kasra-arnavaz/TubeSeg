import pickle

def read_pickle(file):
    with open(file, 'rb') as f:
        out = pickle.load(f)
    return out
