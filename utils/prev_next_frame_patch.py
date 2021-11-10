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
        return [NameParser(name) for name in os.listdir(self.patch_path) if '2015' not in name]

    def movie_names(self):
        return [nameparser.movie_name for nameparser in self.current_frame_patches_nameparsers()]

    def time_points(self):
        return [int(nameparser.time_point.split('-')[0][2:]) for nameparser in self.current_frame_patches_nameparsers()]

    def patch_ids(self):
        return [nameparser.patch_id for nameparser in self.current_frame_patches_nameparsers()]

    def prev_next_frames(self):
        tif_list, name_parsers = [], []
        for movie_name, time_point in zip(self.movie_names(), self.time_points()):
            no_LI_name = movie_name.replace('LI-', '')
            name_with_ = movie_name.replace('LI-', 'LI_').replace('-emb', '_emb').replace('-pos', '_pos')
            name_parsers.append(NameParser(f'pred-0.7-semi-40_ts_{no_LI_name}_tp{time_point+1}_.tif'))
            name_parsers.append(NameParser(f'pred-0.7-semi-40_ts_{no_LI_name}_tp{time_point-1}_.tif'))
            tif_list.append(tif.imread(f"{self.movie_path}/{name_with_}/pred/pred-0.7-semi-40_{name_with_.replace('LI_', '')}_tp{time_point+1}.tif"))
            tif_list.append(tif.imread(f"{self.movie_path}/{name_with_}/pred/pred-0.7-semi-40_{name_with_.replace('LI_', '')}_tp{time_point-1}.tif"))
        return name_parsers, tif_list, np.repeat(self.patch_ids(), 2)
    
    def select_patches(self):
        patch_name_parsers, patches = [], []
        for name_parser, image, patch_id in zip(*self.prev_next_frames()):
            patch_name_parser, patch = ImageOperations(name_parser, image).select_patch(patch_id)
            patch_name_parsers.append(patch_name_parser)
            patches.append(patch)
        return patch_name_parsers, patches

    def write_patches(self):
        for patch_name_parser, patch in zip(*self.select_patches()):
            TifWriter(self.write_path, patch_name_parser, patch).write()


if __name__ == '__main__':
    NeighborPatch('dataset/test/patches/cmp', 'movie/test', 'dataset/test/prev_next_patches_semi').write_patches()
