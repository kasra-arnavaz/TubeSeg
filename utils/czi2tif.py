import tifffile as tif
import czifile as czi
import numpy as np
import os

def czi2tif(czi_path, name, tif_path):
    if not os.path.exists(tif_path):
       x = czi.imread(f'{czi_path}/{name}.czi').squeeze()
       T = x.shape[1]
       for t in range(T):
           os.makedirs(tif_path, exist_ok=True)
           tif.imwrite(f'{tif_path}/{name}_tp{t+1}.tif', x[-2,t])
