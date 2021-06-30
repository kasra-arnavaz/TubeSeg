import numpy as np
import tifffile as tif

x = tif.imread('C:/Users/arnav/Desktop/prob-ae-200_tr_LI-2016-03-04-emb5-pos3_tp70_.tif')
y = tif.imread('C:/Users/arnav/Desktop/ae_r20_s200_tr_prob_LI_2016-03-04_emb5_pos3_tp69.tif')
print(np.array_equal(x,y))
print(np.mean(np.abs(x-y)))