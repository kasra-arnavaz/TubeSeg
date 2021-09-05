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
    v_p, v_l, s_p, s_l = np.clip(v_p, 0, 1), np.clip(v_l, 0, 1), np.clip(s_p, 0,1), np.clip(s_l, 0,1)
    if len(v_p.shape)==3:
        tprec = cl_score(v_p, s_l)
        tsens = cl_score(v_l, s_p)
    if tprec+tsens>0: return 2*tprec*tsens/(tprec+tsens)
    else: return 0
    
def clDice_wrapper(label_path,pred_path, name, thr, model_name, epoch, split):
    v_p = tif.imread(f'{pred_path}/pred-{thr}-{model_name}-{epoch}_{split}_{name}.tif')
    v_l = tif.imread(f'{label_path}/label_{split}_{name}.tif')
    if f'pred-{thr}-{model_name}-{epoch}_{split}_{name}.npy' not in pred_path:
        Skeletonize(pred_path, f'pred-{thr}-{model_name}-{epoch}_{split}_{name}').write_npy()
    if f'label_{split}_{name}.npy' not in label_path:
        Skeletonize(label_path, f'label_{split}_{name}').write_npy()
    s_p = np.load(f'{pred_path}/pred-{thr}-{model_name}-{epoch}_{split}_{name}.npy')
    s_l = np.load(f'{label_path}/label_{split}_{name}.npy')
    return clDice(v_p, v_l, s_p, s_l)

def thr_selection(label_path, pred_path, model_name, epoch):
    names = [name.replace('label_', '').replace('.tif', '').replace(f'val_', '') for name in os.listdir(label_path) if name.startswith('label') and name.endswith('.tif')]
    mean_score_dict, std_score_dict = {}, {}
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 , 0.8, 0.9]:
        score_dict = {}
        for name in names:
            score_dict[name] = clDice_wrapper(label_path, pred_path, name, thr, model_name, epoch, 'val')
        pd.DataFrame.from_dict(score_dict, orient='index').to_csv(f'{pred_path}/clDice_score_thr={thr}.csv')
        thr_score = np.array([score for score in score_dict.values()])
        mean_score_dict[thr] = np.mean(thr_score)
        std_score_dict[thr] = np.std(thr_score)
    
    print(mean_score_dict)
    print(std_score_dict)

def score_thr(label_path, pred_path, model_name, epoch, thr, split):
    names = [name.replace('label_', '').replace('.tif', '').replace(f'{split}_', '') for name in os.listdir(label_path) if name.startswith('label') and name.endswith('.tif')]
    mean_score_dict, std_score_dict = {}, {}
    score_dict = {}
    for name in names:
        print(name)
        score_dict[name] = clDice_wrapper(label_path, pred_path, name, thr, model_name, epoch, split)
    pd.DataFrame.from_dict(score_dict, orient='index').to_csv(f'{pred_path}/clDice_score_thr={thr}.csv')
    thr_score = np.array([score for score in score_dict.values()])
    mean_score_dict[thr] = np.mean(thr_score)
    std_score_dict[thr] = np.std(thr_score)
    
    print(f'cldice_score: {np.mean(thr_score):.3f}±{np.std(thr_score):.3f}')


        


    
if __name__ == '__main__':
    # score_thr('D:/dataset/test/patches/label', 'results/unetcldice/2d/ts/patches', 'unetcldice', 200, 0.5, 'ts')
    score_thr('D:/dataset/test/patches/label', 'results/ae/2d/seg/images/pred/ts/0.7/patches', 'ae', 200, 0.7, 'ts')
    # score_thr('D:/dataset/test/patches/label', 'results/unet/2d/images/pred/ts/0.9/patches', 'unet', 200, 0.9, 'ts')
    # score_thr('D:/dataset/test/patches/label', 'results/semi/2d/images/pred/ts/0.7/patches', 'semi', 40, 0.7, 'ts')
