from models.semi import SemiSupervised
from utils.lsm2tif import lsm2tif
import os
from utils.transform_data import *
import shutil

def main():

    lsm_path = '../raw_data'
    names = [name.replace('.lsm', '') for name in os.listdir(lsm_path) if name.endswith('.lsm')]
#'LI_2019-11-08_emb5_pos3', 'LI_2019-11-08_emb5_pos4',
            #'LI_2020-09-24_emb2_pos1', 
    names = ['LI_2020-09-24_emb2_pos2', 'LI_2016-03-04_emb5_pos3', 'LI_2018-09-28_emb3_pos2', 'LI_2018-09-28_emb5_pos1',
            'LI_2019-02-05_emb5_pos1', 'LI_2019-01-17_emb7_pos3', 'LI_2019-01-17_emb1_pos1',
            'LI_2018-12-18_emb4_pos4', 'LI_2018-12-18_emb4_pos2']
    for name in names:
        tif_path = f'../results/misc/{name}/tif'
        pred_path = f'../results/misc/{name}/pred'
        if not os.path.exists(pred_path):
           print(name)
           lsm2tif(lsm_path, name, tif_path)
           semi = SemiSupervised(name='semi', resume_epoch=40, final_epoch=40, loss_weights=[1, 10], transformer=ModifiedStandardization)
           semi.test_model(tif_path, write_path = f'{tif_path}/..')
           shutil.rmtree(tif_path)

if __name__ == '__main__':
    main()
