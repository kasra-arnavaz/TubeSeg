import numpy as np
import tifffile as tif
import os

def keep_first_T(path, name, ext, T):
    N = 1000
    for t in range(T, N):
        try:
            os.remove(f'{path}/{name}_tp{t+1}.{ext}')
        except:
            pass

if __name__ == '__main__':
    keep_first_T('../results/test-new/LI_2020-07-02_emb4_pos4/cyc', 'pred-0.7-semi-40_2020-07-02_emb4_pos4', 'cyc', 259)
