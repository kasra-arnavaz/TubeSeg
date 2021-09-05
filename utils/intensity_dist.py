import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import os

def intensity_hist(tif_file):

    x = tif.imread(tif_file).reshape(-1)
    plt.hist(x, bins=20)
    plt.show()

def mean_var_intensity_hist(tif_path):

    names = [file for file in os.listdir(tif_path) if file.endswith('.tif')]
    x_mean, x_std = [], []
    for name in names:
        x = tif.imread(f'{tif_path}/{name}')
        x_mean.append(x.mean())
        x_std.append(x.std())
    plt.figure()
    plt.subplot(121)
    plt.hist(x_mean, bins=20)
    plt.subplot(122)
    plt.hist(x_std, bins=20)
    plt.show()


if __name__ == '__main__':
    # intensity_hist('D:/dataset/test/images/duct/duct_ts_LI-2018-11-20-emb7-pos4_tp71_.tif')
    mean_var_intensity_hist('D:/dataset/test/images/duct')
