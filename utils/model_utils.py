import os
import numpy as np
import tifffile as tif


def extract_and_transform_training_images(tr_path, dv_path=None):
    ''' tr_path is where the labeled training images are located. It assumes the labels to have
    an extra 'seg_' as prefix in their names. dv_path is where unlabeled training images are located.
    It returns all those K 2D images in those paths s.t. (y,x).shape = K x 1024 x 1024.
    Elements of y are either 0 or 1 if the image is labeled. If a slice is not labeled, its y is -1. 
    'z_list' contains the number of slices in each 3D image, used later to retrieve the 3D images.
    'names' are the names of the 3D images, used later when saving the probability maps.
     'frac_labeled' is the fraction of 3D labeled images over all, used in semisupervised U-Net to ensure
     labeled images are not outnumbered significantly in each batch.
     'mean' and 'std' are used later to revert the transformation when making the reconstruction of AE.
    '''

    names_tr = [f for f in os.listdir(tr_path) if f.startswith('LI')]
    x = np.zeros([1,1024,1024])
    y = np.zeros([1,1024,1024])
    z_list = []

    for name_tr in names_tr:
        temp_x = tif.imread(f'{tr_path}/{name_tr}')
        temp_y = tif.imread(f'{tr_path}/seg_{name_tr}')
        temp_y = np.clip(temp_y, 0, 1)
        z_list.append(temp_x.shape[0])
        x = np.append(x, temp_x, axis=0)
        y = np.append(y, temp_y, axis=0)

    if dv_path is not None:
        names_dv = [f for f in os.listdir(dv_path) if f.startswith('LI')]
        for name_dv in names_dv:
            temp = tif.imread(f'{dv_path}/{name_dv}')
            z_list.append(temp.shape[0])
            x = np.append(x, temp, axis=0)

    x, y = x[1:], y[1:]
    mean = np.mean(x, axis=(1,2)).reshape(-1,1,1)
    std = np.std(x, axis=(1,2)).reshape(-1,1,1)
    x = x - mean
    x = x / (3*std)
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1) # because the intensity for foreground images are sometimes 255.

    if dv_path is None:
        return x, y, names_tr, z_list
    else:
        neg_one = -1*np.ones((x.shape[0]-y.shape[0], x.shape[1], x.shape[2]))
        y = np.concatenate((y, neg_one), axis=0)
        names = names_tr + names_dv
        
        return x, y, names, z_list, mean, std


def extract_and_transform_test_images(path):
    ''' Similar actions as 'extract_and_transform_training_images' function
    except that x, mean, std are now dictionaries whose keys are the names of 3D images
    and x values are 3D images; mean and std values are their values for each 2D slice.
    So if one 3D image has Z slices x value is of size (Z x 1024 x 1024) and mean and std
    values are of size (Z x 1 x 1).
    '''
	
    names = [f.replace('.tif', '') for f in os.listdir(path) if f.startswith('LI')]
    x, mean, std = {}, {}, {}
    for name in names:
        img = tif.imread(f'{path}/{name}.tif')
        mean[name] = np.mean(img, (1,2)).reshape(-1,1,1)
        std[name] = np.mean(img, (1,2)).reshape(-1,1,1)
        preprocessed_img = (img - mean[name])/(3*std[name])
        x[name] = np.clip(preprocessed_img, 0, 1)

    return x, mean, std


def make_valid_patch(x, name, y=None, input_size=320, output_size=256):
    ''' Makes a random patch from the 1024 x 1024 slice (input x), while avoiding the forbidden regions.
    It assumes the forbiden regions are specified after the last _ in their names.
    This function is used in 'load_training_patches' in 'ParentModel'.
    It returns the selected patch as 'x_patched' of size (input_size x input_size).
    It also returns a corresponding target whose location is centered around x_patched location,
    so target has the size of (output_size x output_size). 
    If x has a label (y is not None) target is the binary label, o.w. the target is image itself but centered.
    '''
    tag = ['A4','B4','C4','D4','A3','B3','C3','D3','A2','B2','C2','D2','A1','B1','C1','D1']
    margin = int((input_size-output_size)/2)
    temp = np.arange(1024**2).reshape(1024,1024)
    d = int(1024/256)
    p = temp.reshape(d,256,d,256).transpose(0,2,1,3)
    patches = p.reshape(-1,256,256)
    for i, loc in enumerate(tag):
        if loc in name: patches[i] = -1
    rec = patches.reshape(d,d,256,256).transpose(0,2,1,3).reshape(1024,1024)
    neg_mask = rec<0
    mar = input_size-1
    idx_neg = np.where(neg_mask.reshape(-1))[0]
    row_neg, col_neg = np.unravel_index(idx_neg,(1024,1024))

    for r,c in zip(row_neg,col_neg):
        if r-mar<0 and c-mar<0:
            rec[0:r+1,0:c+1] = -1
        elif r-mar<0 and c-mar>0:
            rec[0:r+1,c-mar:c+1] = -1
        elif r-mar>0 and c-mar<0:
            rec[r-mar:r+1,0:c+1] = -1
        else:
            rec[r-mar:r+1,c-mar:c+1] = -1
    rec[-mar:, :] = -1
    rec[:, -mar:] =  -1
    rec[-output_size+1:, :] = -1
    rec[:, -output_size+1:] =  -1
    idx_neg = np.where(neg_mask.reshape(-1))[0]
    if len(idx_neg) > 0:
        row_neg = idx_neg//1024
        col_neg = idx_neg%1024
        rn_min, rn_max = np.amin(row_neg), np.amax(row_neg)
        cn_min, cn_max = np.amin(col_neg), np.amax(col_neg)
        rec[rn_min-output_size+1:rn_min, cn_min:cn_max+1] = -1
        rec[rn_min:rn_max+1, cn_min-output_size+1:cn_min] = -1
        rec[rn_min-output_size+1:rn_min+1, cn_min-output_size+1:cn_min+1] = -1
    mask = rec>=0
    idx = np.where(mask.reshape(-1))[0]
    r = np.random.choice(idx, replace=True)
    r_h = r//1024
    r_w = r%1024
    x_pad = np.pad(x, ((margin,margin),(margin,margin)), 'constant', constant_values=0)
    x_patched = x_pad[r_h:r_h+input_size, r_w:r_w+input_size]

    if y is None:
        x_hat = x[r_h:r_h+output_size, r_w:r_w+output_size]
        return x_patched, x_hat
    else:
        y_patched = y[r_h:r_h+output_size, r_w:r_w+output_size]
        return x_patched, y_patched

if __name__ == '__main__':
    name = 'prob-semi-40_val_LI-2016-03-04-emb5-pos2_tp49-A1A2A3A4_.tif'
    path = 'utils/unittest_data/prob'
    x = tif.imread(f'{path}/{name}')[0]
    for i in range(100):
        valid = make_valid_patch(x, name)[0]
    # tif.imwrite(f'{path}/asdf.tif', valid)
    