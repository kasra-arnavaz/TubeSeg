import tifffile as tif
import czifile as czi
import numpy as np
import os

def lsm2tif(czi_path, name, tif_path):
    if not os.path.exists(tif_path):
       x = tif.imread(f'{lsm_path}/{name}.lsm').squeeze()
       T = x.shape[1]
       for t in range(T):
           os.makedirs(tif_path, exist_ok=True)
           tif.imwrite(f'{tif_path}/{name}_tp{t+1}.tif', x[-2,t])
