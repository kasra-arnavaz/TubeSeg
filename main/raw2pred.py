
import os
import shutil
from argparse import ArgumentParser

from models.semi import SemiSupervised
from utils.raw2tif import czi2tif, lsm2tif
from utils.transform_data import ModifiedStandardization

def raw2pred(raw_file, save_to, tp_max=None, remove_LI=True):

   movie_name = raw_file.split('/')[-1][:-4]
   LI_path = f'{save_to}/{movie_name}/LI'
   if not os.path.exists(LI_path):
      if raw_file.endswith('.lsm'): lsm2tif(raw_file, LI_path, tp_max)
      elif raw_file.endswith('.czi'): czi2tif(raw_file, LI_path, tp_max)
      else: ValueError(f'{raw_file} should be an lsm or czi file format.')
   if not os.path.exists(f'{save_to}/{movie_name}/pred'):
      semi = SemiSupervised(name='semi', resume_epoch=40, final_epoch=40, loss_weights=[1, 10], transformer=ModifiedStandardization)
      semi.test_model(LI_path, write_path = f'{LI_path}/..')
   if remove_LI: shutil.rmtree(LI_path)
        

if __name__ == '__main__':
   parser = ArgumentParser()
   parser.add_argument('--raw_file', type=str)
   parser.add_argument('--save_to', type=str)
   parser.add_argument('--tp_max', type=int, default=None)
   parser.add_argument('--remove_LI', type=bool, default=True)
   args = parser.parse_args()
   raw2pred(args)
