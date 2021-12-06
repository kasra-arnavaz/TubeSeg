import tifffile as tif
import czifile as czi
import numpy as np
import os

def czi2tif(czi_file, tif_path, until_tp=None):
    if not os.path.exists(tif_path):
       x = czi.imread(czi_file).squeeze()
       name = czi_file.split('/')[-1].replace('.czi', '')
       if until_tp is not None: tp_max = x.shape[1]
       for t in range(tp_max):
           os.makedirs(tif_path, exist_ok=True)
           tif.imwrite(f'{tif_path}/{name}_tp{t+1}.tif', x[-2,t])


def lsm2tif(lsm_file, tif_path, until_tp=None):
    name = lsm_file.split('/')[-1].replace('.lsm', '')
    with tif.TiffFile(lsm_file) as f:
        series = f.series[0]
        t, z, c = series.shape[:3]
        xt = np.zeros((series.shape[1:]), dtype=np.int16)
        tp = 0
        if until_tp is not None: tp_max = until_tp
        for i, page in enumerate(series):
            xt[i%z] = page.asarray()
            if (i%z == (z-1)):
                tp +=1
                if tp > tp_max: break
                os.makedirs(tif_path, exist_ok=True)
                tif.imwrite(f"{tif_path}/{name}_tp{tp}.tif", xt[:,-2])