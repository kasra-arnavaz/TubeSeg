# from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import tifffile as tif
import os
import pandas as pd

from utils.skel2binary import Skeletonize

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l, s_p, s_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    # if len(v_p.shape)==2:
    #     tprec = cl_score(v_p,skeletonize(v_l))
    #     tsens = cl_score(v_l,skeletonize(v_p))
    if len(v_p.shape)==3:
        tprec = cl_score(v_p, s_l)
        tsens = cl_score(v_l, s_p)
    return 2*tprec*tsens/(tprec+tsens)

def clDice_wrapper(path, name, thr, model_name, epoch, split):
    v_p = tif.imread(f'{path}/pred-{thr}-{model_name}-{epoch}_{split}_{name}.tif')
    v_l = tif.imread(f'{path}/label_{split}_{name}.tif')
    if f'pred-{thr}-{model_name}-{epoch}_{split}_{name}.npy' not in path:
        Skeletonize(path, f'pred-{thr}-{model_name}-{epoch}_{split}_{name}').write_npy()
    if f'label_{split}_{name}.npy' not in path:
        Skeletonize(path, f'label_{split}_{name}').write_npy()
    s_p = np.load(f'{path}/pred-{thr}-{model_name}-{epoch}_{split}_{name}.npy')
    s_l = np.load(f'{path}/label_{split}_{name}.npy')
    return clDice(v_p, v_l, s_p, s_l)

def thr_selection(path, model_name, epoch):
    names = [name.replace('label_', '').replace('.tif', '').replace(f'val_', '') for name in os.listdir(path) if name.startswith('label') and name.endswith('.tif')]
    mean_score_dict, std_score_dict = {}, {}
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 , 0.8, 0.9]:
        score_dict = {}
        for name in names:
            score_dict[name] = clDice_wrapper(path, name, thr, model_name, epoch, 'val')
        pd.DataFrame.from_dict(score_dict, orient='index').to_csv(f'{path}/clDice_score_thr={thr}.csv')
        thr_score = np.array([score for score in score_dict.values()])
        mean_score_dict[thr] = np.mean(thr_score)
        std_score_dict[thr] = np.std(thr_score)
    
    print(mean_score_dict)
    print(std_score_dict)


        


    
if __name__ == '__main__':
    thr_selection('results/unet/2d/images/pred/val/patches', 'unet', 200)
