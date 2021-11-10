import tifffile as tif
import os
import numpy as np

def lsm2tif(lsm_dir):
    names = [name.replace('.lsm', '') for name in os.listdir(lsm_dir) if name.endswith('.lsm')]
    for i, name in enumerate(names):
        print(f"{i}/{len(names)}")
        os.makedirs(f"{lsm_dir}/{name}/gfp", exist_ok=True)
        os.makedirs(f"{lsm_dir}/{name}/mcherry", exist_ok=True)
        with tif.TiffFile(f'{lsm_dir}/{name}.lsm') as f:
            series = f.series[0]
            t, z, c = series.shape[:3]
            xt = np.zeros((series.shape[1:]), dtype=np.int16)
            tp = 0
            for i, page in enumerate(series):
                xt[i%z] = page.asarray()
                if (i%z == (z-1)):
                    tp +=1
                    if c == 3:
                        tif.imwrite(f"{lsm_dir}/{name}/gfp/{name}_tp{tp}.tif", xt[:,0])
                    tif.imwrite(f"{lsm_dir}/{name}/mcherry/{name}_tp{tp}.tif", xt[:,-2])
                    tif.imwrite(f"{lsm_dir}/{name}/mcherry/{name}_tp{tp}.tif", xt[:,-2])
                    

