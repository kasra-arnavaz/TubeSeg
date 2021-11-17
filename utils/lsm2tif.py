import tifffile as tif
import os
import numpy as np

def lsm2tif(lsm_dir, name, tif_dir):
    with tif.TiffFile(f'{lsm_dir}/{name}.lsm') as f:
        series = f.series[0]
        t, z, c = series.shape[:3]
        xt = np.zeros((series.shape[1:]), dtype=np.int16)
        tp = 0
        for i, page in enumerate(series):
            xt[i%z] = page.asarray()
            if (i%z == (z-1)):
                tp +=1
                os.makedirs(tif_dir, exist_ok=True)
                tif.imwrite(f"{tif_dir}/{name}_tp{tp}.tif", xt[:,-2])
                    

