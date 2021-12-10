import numpy as np
import os


def make_valid_patch(x, name, y=None, input_size=320, output_size=256, image_size=1024):
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
    temp = np.arange(image_size**2).reshape(image_size,image_size)
    d = int(image_size/output_size)
    p = temp.reshape(d,output_size,d,output_size).transpose(0,2,1,3)
    patches = p.reshape(-1,output_size,output_size)
    for i, loc in enumerate(tag):
        if loc in name: patches[i] = -1
    rec = patches.reshape(d,d,output_size,output_size).transpose(0,2,1,3).reshape(image_size,image_size)
    neg_mask = rec<0
    mar = input_size-1
    idx_neg = np.where(neg_mask.reshape(-1))[0]
    row_neg, col_neg = np.unravel_index(idx_neg,(image_size,image_size))

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
        row_neg = idx_neg//image_size
        col_neg = idx_neg%image_size
        rn_min, rn_max = np.amin(row_neg), np.amax(row_neg)
        cn_min, cn_max = np.amin(col_neg), np.amax(col_neg)
        rec[rn_min-output_size+1:rn_min, cn_min:cn_max+1] = -1
        rec[rn_min:rn_max+1, cn_min-output_size+1:cn_min] = -1
        rec[rn_min-output_size+1:rn_min+1, cn_min-output_size+1:cn_min+1] = -1
    mask = rec>=0
    idx = np.where(mask.reshape(-1))[0]
    r = np.random.choice(idx, replace=True)
    r_h = r//image_size
    r_w = r%image_size
    x_pad = np.pad(x, ((margin,margin),(margin,margin)), 'constant', constant_values=0)
    x_patched = x_pad[r_h:r_h+input_size, r_w:r_w+input_size]

    if y is None:
        x_hat = x[r_h:r_h+output_size, r_w:r_w+output_size]
        return x_patched, x_hat
    else:
        y_patched = y[r_h:r_h+output_size, r_w:r_w+output_size]
        return x_patched, y_patched


def convert_patches_into_image(patches):
    ''' patches (Z x 16 x 256 x 256) --> image (Z x 1024 x 1024)
    '''
    return patches.reshape(-1,4,4,256,256).transpose([0,1,3,2,4]).reshape([-1,1024,1024])
