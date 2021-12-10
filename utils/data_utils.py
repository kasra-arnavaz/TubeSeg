import os
import numpy as np
import tifffile as tif


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

# write_patches_from_images('E:/dataset/val/patches/duct/320', 'E:/dataset/val/images/duct', 'E:/dataset/val/patches/duct')

def make_patches(saving_path, target_path, pred_path, prefix, epoch, thr_list):

    names_target = [f.replace('label_','').replace('.tif' ,'') for f in os.listdir(target_path) if f.endswith('tif') ]
    for thr in thr_list:
        for name in names_target:
            patch_id = name[-2:]
            raw_name = name.replace(patch_id, '')
            # y = tif.imread(f'{pred_path}/prob-{prefix}-{epoch}_{raw_name}.tif')
            y = tif.imread(f'{pred_path}/pred-{thr}-{prefix}-{epoch}_{raw_name}.tif')
            y = y.reshape(-1,4,256,4,256).transpose(0,1,3,2,4)
            y = y.reshape(-1,16,256,256).transpose(1,0,2,3,)
            patch_index = get_patch_index(patch_id)
            y = y[patch_index]
            os.makedirs(saving_path, exist_ok=True)
            tif.imwrite(f'{saving_path}/pred-{thr}-{prefix}-{epoch}_{raw_name}{patch_id}.tif', y)
            # tif.imwrite(f'{saving_path}/prob-{prefix}-{epoch}_{raw_name}{patch_id}.tif', y)

def make_patches_320(saving_path, patch_path, image_path):

    names_patch = [f.replace('label_','').replace('.tif' ,'') for f in os.listdir(patch_path) if f.endswith('tif') ]
    in_size, out_size = 320, 256
    dh_size = int((in_size - out_size)/2)
    for name in names_patch:
        patch_id = name[-2:]
        raw_name = name.replace(patch_id, '')
        patch_index = get_patch_index(patch_id)
        r, c = np.unravel_index(patch_index,(4,4))
        x_test = tif.imread(f'{image_path}/{raw_name}.tif')
        x_test_pad = np.pad(x_test, ((0,0),(dh_size,dh_size),(dh_size,dh_size)), 'constant', constant_values=0)
        r *= out_size
        c *= out_size
        x_que = x_test_pad[:,r:r+in_size,c:c+in_size]
        os.makedirs(saving_path, exist_ok=True)
        tif.imwrite(f'{saving_path}/{name}.tif', x_que)

def make_patches_movies(patch_path, movie_path, prefix, thr, epoch):

    for movie_name in os.listdir(movie_path):
        patches = [name[-6:-4] for name in os.listdir(patch_path) if (name.endswith('.tif')) and (movie_name.replace('_','-') in name)]
        raw_name = movie_name.replace('LI_', '')
        tif_preds = [name for name in os.listdir(f'{movie_path}/{movie_name}/pred') if name.endswith('.tif')]
        print(tif_preds)
        for patch in patches:
            for t in range(len(tif_preds)):
                print(patch, t)
                y = tif.imread(f'{movie_path}/{movie_name}/pred/pred-{thr}-{prefix}-{epoch}_{raw_name}_tp{t+1}.tif')
                y = y.reshape(-1,4,256,4,256).transpose(0,1,3,2,4)
                y = y.reshape(-1,16,256,256).transpose(1,0,2,3,)
                patch_index = get_patch_index(patch)
                y = y[patch_index]
                saving_path = f'{movie_path}/{movie_name}/pred_patches/{patch}'
                os.makedirs(saving_path, exist_ok=True)
                tif.imwrite(f'{saving_path}/pred-{thr}-{prefix}-{epoch}_{raw_name}_tp{t+1}_{patch}.tif', y)

def get_patch_index(patch_id):
    grid = np.array(['A4','B4','C4','D4',
                        'A3','B3','C3','D3',
                        'A2','B2','C2','D2',
                        'A1','B1','C1','D1'])
    return int(np.argwhere(grid==patch_id))


if __name__ == '__main__':
    make_patches('results/semi/2d/images/pred/ts/0.7/patches', 'D:/dataset/test/patches/label', f'results/semi/2d/images/pred/ts/0.7', 'semi', 40, [0.7])