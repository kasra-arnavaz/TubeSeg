from models.semi import SemiSupervised
from utils.lsm2tif import lsm2tif
import os
from utils.transform_data import *
import shutil

def main():

    lsm_path = '../raw_data'
    names = [name.replace('.lsm', '') for name in os.listdir(lsm_path) if name.endswith('.lsm')]
    for name in names:
        tif_path = f'../results/{name}/tif'
        pred_path = f'../results/{name}/pred'
        if not os.path.exists(pred_path):
           print(name)
           lsm2tif(czi_path, name, tif_path)
           semi = SemiSupervised(name='semi', resume_epoch=40, final_epoch=40, loss_weights=[1, 10], transformer=ModifiedStandardization)
           semi.test_model(tif_path, write_path = f'{tif_path}/..')
        shutil.rmtree(tif_path)

if __name__ == '__main__':
    main()
