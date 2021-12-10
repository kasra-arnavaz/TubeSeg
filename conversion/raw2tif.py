import tifffile as tif
import czifile as czi
import numpy as np
import os
import warnings
from argparse import ArgumentParser

def czi2tif(czi_file: str, write_path: str, make_duct: bool, make_beta: bool, tp_max: int = None) -> None:
    ''' Writes 4D tif files for every frame.
        czi_file: path_like referring to czi_file e.g. './raw_data/LI_2018-05-10_emb4_pos3.czi'.
        write_path: path to write the tif files to e.g. duct(or beta) files are saved to 'write_path/LI_2018-05-10_emb4_pos3/duct'.
        make_duct: should duct channel be written.
        make_beta: should beta channel be written.
        tp_max: if specified, timepoints after tp_max are not written.
    '''

    if make_duct or make_beta:  
        movie_name = czi_file.split('/')[-1].replace('.czi', '')
        duct_path = f'{write_path}/{movie_name}/duct'
        beta_path = f'{write_path}/{movie_name}/beta'
        x = czi.imread(czi_file).squeeze()
        if tp_max > x.shape[1]: raise ValueError(f'tp_max exceeds the number of frames in the raw data.')
        if tp_max is not None: tp_max = x.shape[1]
        for t in range(tp_max):
            if make_duct:
                os.makedirs(duct_path, exist_ok=True)
                tif.imwrite(f'{duct_path}/duct_{movie_name}_tp{t+1}_.tif', x[-2,t])
            if make_beta and x.shape[0] == 3:
                os.makedirs(beta_path, exist_ok=True)
                tif.imwrite(f'{beta_path}/beta_{movie_name}_tp{t+1}_.tif', x[0,t])
            elif make_beta and x.shape[0] != 3:
                warnings.warn(f'{czi_file} does not have a beta-cell channel.')
    else: warnings.warn('Did nothing; make_duct and make_beta were both set to False!')


def lsm2tif(lsm_file: str, write_path: str, make_duct: bool, make_beta: bool, tp_max: int = None) -> None:
    ''' Writes 4D tif files for every frame.
    lsm_file: path_like referring to lsm_file e.g. './raw_data/LI_2018-05-10_emb4_pos3.lsm'.
    write_path: path to write the tif files to e.g. duct(or beta) files are saved to 'write_path/LI_2018-05-10_emb4_pos3/duct'.
    make_duct: should duct channel be written.
    make_beta: should beta channel be written.
    tp_max: if specified, timepoints after tp_max are not written.
    '''

    if make_duct or make_beta: 
        movie_name = lsm_file.split('/')[-1].replace('.lsm', '')
        duct_path = f'{write_path}/{movie_name}/duct'
        beta_path = f'{write_path}/{movie_name}/beta'
        with tif.TiffFile(lsm_file) as f:
            series = f.series[0]
            t, z, c = series.shape[:3]
            if tp_max is None: tp_max = t
            elif tp_max > t: raise ValueError(f'tp_max exceeds the number of frames in the raw data.')
            xt = np.zeros((series.shape[1:]), dtype=np.int16)
            tp = 0
            for i, page in enumerate(series):
                xt[i%z] = page.asarray()
                if (i%z == (z-1)):
                    tp +=1
                    if tp > tp_max: break
                    if make_duct:
                        os.makedirs(duct_path, exist_ok=True)
                        tif.imwrite(f'{duct_path}/duct_{movie_name}_tp{tp}_.tif', xt[:,-2])
                    if (make_beta) and (c == 3):
                        os.makedirs(beta_path, exist_ok=True)
                        tif.imwrite(f'{beta_path}/beta_{movie_name}_tp{tp}_.tif', xt[:,0])
                    elif make_beta and c != 3:
                        warnings.warn(f'{lsm_file} does not have a beta-cell channel.')
    else: warnings.warn('Did nothing; make_duct and make_beta were both set to False!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--raw_file', type=str)
    parser.add_argument('--write_path', type=str)
    parser.add_argument('--make_duct', action='store_true')
    parser.add_argument('--make_beta', action='store_true')
    parser.add_argument('--tp_max', type=int)
    args = parser.parse_args()
    if args.raw_file.endswith('.czi'):
        czi2tif(args.raw_file, args.write_path, args.make_duct, args.make_beta, args.tp_max)
    elif args.raw_file.endswith('.lsm'):
        lsm2tif(args.raw_file, args.write_path, args.make_duct, args.make_beta, args.tp_max)
    else: raise ValueError(f'{args.raw_file} should be of lsm or czi file format.')

