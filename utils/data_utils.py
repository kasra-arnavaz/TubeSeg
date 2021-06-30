import os
import numpy as np
import tifffile as tif

# class Name:

#     def __init__(self, path):
#         self.path = path
    
#     def separate_on_extension(self, name):
#         name_without_ext, ext = os.path.splitext(name)
#         return name_without_ext, ext
    
#     def is_tif(self, name):
#         return self.separate_on_extension(name)[1] == '.tif'
    
#     def exclude_category_and_extension(self, name):
#         split_name = name.split('_')
#         return '_'.join(split_name[1:])

#     @property
#     def get_names(self):
#         '''tif names without category and extension'''
#         return [self.exclude_category_and_extension(name) for name in os.listdir(self.path) if self.is_tif(name)]
            
class NameParser:
    '''
    category         starts with e.g. duct, label, prob, pred, ent
    split            e.g. tr, dev, val, ts
    movie_name       e.g. LI-2020-10-10-emb3-pos2
    time_point       e.g. tp13-D3D4C4
    extension        e.g. tif, png, pdf
    patch_id         '' for image, e.g. A2 for patch
    '''    
    def __init__(self, name):
        rest, self.extension = os.path.splitext(name)
        self.category, self.split, self.movie_name, self.time_point, self.patch_id = rest.split('_')
        
    @property
    def name(self):
        name = '_'.join([self.category, self.split, self.movie_name, self.time_point, self.patch_id])
        return f'{name}{self.extension}'
    
    @property
    def frame_name(self):
        return f'{self.movie_name}_{self.time_point}'

    @property
    def separate_name_and_patch_id(self):
        patch_id = self.patch_id
        self.patch_id = ''
        return (self.name, patch_id)

    def change_extension(self, new_extension):
        self.extension = new_extension
        return self
    
    def change_category(self, new_category):
        self.category = new_category
        return self
        



class TifReader:

    def __init__(self, path, name_parser=NameParser):
        self.path = path
        self.name_parser = name_parser
        self.i = 0

    def read(self, name):
        return self.name_parser(name), tif.imread(f'{self.path}/{name}')

    @property
    def names(self):
        return [name for name in os.listdir(self.path) if name.endswith('.tif')]

    @property
    def name_parsers(self):
        return [self.name_parser(name) for name in os.listdir(self.path) if name.endswith('.tif')]

    @property
    def frame_names(self):
        return [self.name_parser(name).frame_name for name in self.names]

    @property
    def separate_names_and_patch_ids(self):
        return [self.name_parser(name).separate_name_and_patch_id for name in self.names]

    def __iter__(self):
        return self
    
    def __next__(self):
        names = self.names
        if self.i < len(names):
            name = names[self.i]
            name_parser, x = self.read(name)
            self.i += 1
            return name_parser, x
        else:
            raise StopIteration

class TifWriter:

    def __init__(self, path, name_parser: NameParser, pic: np.ndarray):
        self.path = path
        self.name_parser = name_parser
        self.pic = pic

    def write(self):
        os.makedirs(f'{self.path}', exist_ok=True)
        tif.imwrite(f'{self.path}/{self.name_parser.name}', self.pic)



class ImageOperations:

    def __init__(self, name_parser: NameParser, image: np.ndarray):
        self.name_parser = name_parser
        self.image = image

    @staticmethod
    def get_patch_index(patch_id):
        grid = np.array(['A4','B4','C4','D4',
                        'A3','B3','C3','D3',
                        'A2','B2','C2','D2',
                        'A1','B1','C1','D1'])
        return int(np.argwhere(grid==patch_id))


    def convert_image_into_patches(self):
        ''' image (Z, 1024, 1024) --> patches (Z, 16, 256, 256)
        ''' 
        return self.image.reshape(-1,4,256,4,256).transpose(0,1,3,2,4).reshape(-1,16,256,256)


    def select_patch(self, patch_id):
        patches = self.convert_image_into_patches()
        patch_index = self.get_patch_index(patch_id)
        selected_patch = patches[:, patch_index]
        self.name_parser.patch_id = patch_id 
        return self.name_parser, selected_patch

class PatchesOperations:
    
    def __init__(self, name_parser: NameParser, patches: np.ndarray):
        self.name_parser = name_parser
        self.patches = patches

    def patches_are_from_images(self, images):
        images_frame_names = TifReader(images_path).frame_names
        patches_frame_names = TifReader(patches_path).frame_names
        return set(images_frame_names) == set(patches_frame_names)  


def write_patches_from_images(write_dir, images_dir, patches_dir):
    # if  not patches_come_from_images(images_dir, patches_dir): 
    #     raise ValueError(f"Patches in {patches_dir} don't come from images in {images_dir}!")
    for image_name, patch_id in TifReader(patches_dir).separate_names_and_patch_ids:
        crop_320(write_dir, images_dir, image_name, patch_id)
        
def crop_320(write_dir, image_dir, image_name, patch_id):
    in_size = 320
    out_size = 256
    dh_size = int((in_size - out_size)/2)
    tag = ['A4','B4','C4','D4','A3','B3','C3','D3','A2','B2','C2','D2','A1','B1','C1','D1']
    for idx,i in enumerate(tag):
        if i==patch_id: break
    r, c = np.unravel_index(idx,(4,4))
    x_test = tif.imread(f'{image_dir}/{image_name}')
    x_test_pad = np.pad(x_test, ((0,0),(dh_size,dh_size),(dh_size,dh_size)), 'constant', constant_values=0)
    r *= out_size
    c *= out_size
    x_que = x_test_pad[:,r:r+in_size,c:c+in_size]
    os.makedirs(write_dir, exist_ok=True)
    tif.imwrite(f"{write_dir}/{image_name.replace('.tif', '')}{patch_id}.tif", x_que)

write_patches_from_images('E:/dataset/val/patches/duct/320', 'E:/dataset/val/images/duct', 'E:/dataset/val/patches/duct')

# def make_patches(target_path, pred_path, prefix, split, epoch, thr):

#     names_target = [f.replace('seg_','') for f in os.listdir(target_path) if f.endswith('tif') ]

#     for name in names_target:
#         patch_id = name.split('_')[-1][:2]
#         raw_name = name.replace('_'+patch_id, '')
#         t = tif.imread(f'{target_path}/seg_{name}').reshape(-1)
#     #    y = tif.imread(f'{pred_path}/{prefix}_{epoch}_{split}_pred{np.round(thr, 2)}_{raw_name}')
#         y = tif.imread(f'{pred_path}/{prefix}_{epoch}_{split}_lin_{raw_name}')
#         y = y.reshape(-1,4,256,4,256).transpose(0,1,3,2,4)
#         y = y.reshape(-1,16,256,256).transpose(1,0,2,3,)
#         patch_index = get_patch_index(patch_id)
#         y = y[patch_index]
#     #    saving_path  = pred_path.replace('preds', 'preds_patches')
#         saving_path  = pred_path.replace('lins', 'lins_patches')
#         os.makedirs(saving_path, exist_ok=True)
#         tif.imwrite(f'{saving_path}/{prefix}_{epoch}_{split}_lin_{name}', y, 'minisblack')
#     #    tif.imwrite(f'{saving_path}/{prefix}_{epoch}_{split}_pred{np.round(thr, 2)}_{name}', y, 'minisblack')

if __name__ == '__main__':
    print(Name('alaki').get_names)