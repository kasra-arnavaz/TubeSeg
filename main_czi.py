from models.semi import SemiSupervised
from utils.czi2tif import czi2tif
import os
from utils.transform_data import *
import shutil

def main():

    czi_path = '../raw_data/new_czi/sus'
    names = [name.replace('.czi', '') for name in os.listdir(czi_path) if name.endswith('.czi')]
    for name in names:
        tif_path = f'../results/new_czi/{name}/tif'
        if not os.path.exists(tif_path):
           czi2tif(czi_path, name, tif_path)
        if not os.path.exists(f'../results/new_czi/{name}/pred'):
           semi = SemiSupervised(name='semi', resume_epoch=40, final_epoch=40, loss_weights=[1, 10], transformer=ModifiedStandardization)
           semi.test_model(tif_path, write_path = f'{tif_path}/..')
        shutil.rmtree(tif_path)
        

if __name__ == '__main__':
    main()
