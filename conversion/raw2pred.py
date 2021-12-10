
import os
import shutil
from argparse import ArgumentParser

from segmentation.models.semi import SemiSupervised
from conversion.raw2tif import czi2tif, lsm2tif
from segmentation.utils.transform_data import ModifiedStandardization

def raw2pred(raw_file: str, save_to: str, tp_max: int = None, remove_duct: bool = True) -> None:
   ''' Makes predicted segmentations from raw lsm or czi files for every time point.
   raw_file: path_like referring to lsm or czi files e.g. './raw_data/LI_2018-05-10_emb4_pos3.czi'.
   save_to: path to write the tif files to e.g. duct(or beta) files are saved to 'save_to/LI_2018-05-10_emb4_pos3/duct'.
   tp_max: if specified, timepoints after tp_max are not written.
   remove_duct: if True, duct files would be deleted after making the prediction to save space.
   '''

   movie_name = raw_file.split('/')[-1][:-4]
   if raw_file.endswith('.lsm'): lsm2tif(raw_file, save_to, make_duct=True, make_beta=False, tp_max=tp_max)
   elif raw_file.endswith('.czi'): czi2tif(raw_file, save_to, make_duct=True, make_beta=False, tp_max=tp_max)
   else: ValueError(f'{raw_file} should be an lsm or czi file format.')
   semi = SemiSupervised(name='semi', resume_epoch=40, final_epoch=40, loss_weights=[1, 10], transformer=ModifiedStandardization)
   semi.test_model(f'{save_to}/{movie_name}/duct', write_path = f'{save_to}/{movie_name}')
   if remove_duct: shutil.rmtree(f'{save_to}/{movie_name}/duct')
        

if __name__ == '__main__':
   parser = ArgumentParser()
   parser.add_argument('--raw_file', type=str)
   parser.add_argument('--save_to', type=str)
   parser.add_argument('--tp_max', type=int, default=None)
   parser.add_argument('--remove_LI', action='store_true')
   args = parser.parse_args()
   raw2pred(args.raw_file, args.save_to, args.tp_max, args.remove_LI)
