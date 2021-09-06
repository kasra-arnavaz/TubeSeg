import tifffile as tif
import numpy as np
import os
from utils.data_utils import NameParser, ImageOperations, TifWriter


class NeighborPatch:

    def __init__(self, patch_path, movie_path, write_path):
        self.patch_path = patch_path
        self.movie_path = movie_path
        self.write_path = write_path
    
    def current_frame_patches_nameparsers(self):
        return [NameParser(name) for name in os.listdir(self.patch_path)]

    def movie_names(self):
        return [nameparser.movie_name for nameparser in self.current_frame_patches_nameparsers()]

    def time_points(self):
        return [int(nameparser.time_point.split('-')[0][2:]) for nameparser in self.current_frame_patches_nameparsers()]

    def patch_ids(self):
        return [nameparser.patch_id for nameparser in self.current_frame_patches_nameparsers()]

    def prev_next_frames(self):
        tif_list, name_parsers = []
        for movie_name, time_point in zip(self.movie_names, self.time_points):
            no_LI_name = movie_name.replace('LI', '')
            tif_list.append(f'{self.movie_path}/{movie_name}/pred/pred-0.7-semi-40{no_LI_name}_tp{time_point+1}.tif')
            tif_list.append(f'{self.movie_path}/{movie_name}/pred/pred-0.7-semi-40{no_LI_name}_tp{time_point-1}.tif')
            name_parsers.append(NameParser(f'pred-0.7-semi-40{no_LI_name}_tp{time_point+1}.tif'))
            name_parsers.append(NameParser(f'pred-0.7-semi-40{no_LI_name}_tp{time_point-1}.tif'))

        return tif_list, name_parsers, np.repeat(self.patch_ids, 2)
    
    def select_patches(self):
        for name_parser, image, patch_id in zip(*self.prev_next_frames()):
            yield ImageOperations(name_parser, image).select_patch(patch_id)
    
    def write_patches(self):
        while True:
            TifWriter(self.write_path, *self.select_patches()).write()


if __name__ == '__main__':
    NeighborPatch('dataset/test/patches/cmp', 'movie/test', 'dataset/test/prev_next_patches_semi')