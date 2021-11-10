import os

def rename_images(path):
    names = [name.replace('.tif', '') for name in os.listdir(path) if name.endswith('tif')]
    for old_name in names:
        new_name = old_name.replace('_', '-').replace('-tp', '_tp') + '_'
        os.rename(f'{path}/{old_name}.tif', f'{path}/{new_name}.tif')

def rename_patches(path):
    names = [name.replace('.tif', '') for name in os.listdir(path) if name.endswith('tif')]
    for old_name in names:
        new_name = old_name.replace('_', '-').replace('-tp', '_tp')
        new_name_list = list(new_name)
        new_name_list[-3] = '_'
        new_name = ''.join(new_name_list)
        os.rename(f'{path}/{old_name}.tif', f'{path}/{new_name}.tif')

def rename_labels(path):
    names = [name.replace('.tif', '') for name in os.listdir(path) if name.endswith('tif')]
    for old_name in names:
        new_name = old_name.replace('seg_', '').replace('_', '-').replace('-tp', '_tp')
        new_name_list = list(new_name)
        new_name_list[-3] = '_'
        new_name = ''.join(new_name_list)
        os.rename(f'{path}/{old_name}.tif', f'{path}/{new_name}.tif')

def add_prefix(path, prefix):
    names = [name.replace('.tif', '') for name in os.listdir(path) if name.endswith('tif')]
    for old_name in names:
        new_name = f'{prefix}_{old_name}'
        os.rename(f'{path}/{old_name}.tif', f'{path}/{new_name}.tif')

def rename_graphs(path):
    names = [name.replace('.graph', '') for name in os.listdir(path) if name.endswith('.graph')]
    for old_name in names:
        new_name = old_name.replace('_', '-').replace('-tp', '_tp').replace('skel-', 'skel_')
        new_name_list = list(new_name)
        new_name_list[-3] = '_'
        new_name = ''.join(new_name_list)
        os.rename(f'{path}/{old_name}.graph', f'{path}/{new_name}.graph')

def rename_movies(path):
    for name in os.listdir(path):
        new_name = name.replace('LI_', 'LI-').replace('_emb', '-emb').replace('_pos', '-pos')
        os.rename(f'{path}/{name}', f'{path}/{new_name}')

def rename_skel(path):
    names = [name for name in os.listdir(path) if name.endswith('graph')]
    for name in names:
        new_name = name.replace('skel_', '').replace('.graph', '.skel')
        os.rename(f'{path}/{name}', f'{path}/{new_name}')

def tif_rename(path):
    for name in os.listdir(path):
        
        new_name1 = name.replace('pred0.', 'pred-0.').replace('one-ten_40_val_', '').replace('_LI', '-semi-40_val_LI')
        new_name2 = new_name1.replace('_', '-').replace('-LI', '_LI').replace('-tp', '_tp').replace('-val', '_val')
        list_new = list(new_name2)
        list_new[-7] = '_'
        new_name = ''.join(list_new)
        print(new_name)
        # os.rename(f'{path}/{name}', f'{path}/{new_name}')

def skel_rename(path):
    for name in os.listdir(path):
        
        new_name1 = name.replace('pred0.', 'pred-0.').replace('one-ten_40_val_', '').replace('_LI', '-semi-40_val_LI')
        new_name2 = new_name1.replace('_', '-').replace('-LI', '_LI').replace('-tp', '_tp').replace('-val', '_val').replace('skel-','skel_')
        list_new = list(new_name2)
        list_new[-7] = '_'
        new_name = ''.join(list_new)
        print(new_name)
        os.rename(f'{path}/{name}', f'{path}/{new_name}')

def val_tp_rename(path):
    for name in os.listdir(path):
        new_name = name.replace('2020-05-06-pos4', '2020-05-06-emb7-pos4').replace('embXX', 'embX')
        new_tp_name = new_name.replace('tp105-', 'tp106-').replace('tp264-', 'tp265-').replace('tp12-', 'tp13-').replace('tp79-', 'tp80-').replace('tp76-', 'tp77-').replace('tp87-', 'tp88-').replace('tp231-', 'tp232-')
        print(new_tp_name)
        os.rename(f'{path}/{name}', f'{path}/{new_tp_name}')

def remove_skel(path):
    for name in os.listdir(path):
        new_name = name.replace('skel_', '')
        os.rename(f'{path}/{name}', f'{path}/{new_name}')

def rename_m2(path):
    for name in os.listdir(path):
        new_name = name.replace('val_', 'ts_')
        # print(new_name)
        os.rename(f'{path}/{name}', f'{path}/{new_name}')

def remove_ignored_patches(path):
    i = 0
    for name in os.listdir(path):
        if 'tp' in name.split('-')[-2]:
            i +=1
            ignored_patches = name.split('-')[-1].split('_')[0]
            new_name = name.replace(f'-{ignored_patches}', '')
            # print(name, new_name, i)
            os.rename(f'{path}/{name}', f'{path}/{new_name}')

def remove_LI(path):
    for name in os.listdir(path):
        new_name = name.replace('LI-', '')
        print(name, new_name)
        os.rename(f'{path}/{name}', f'{path}/{new_name}')

path = 'D:/dataset/prev_next_patches_semi'
# tif_rename(path)
# skel_rename(path)
# val_tp_rename(path)
remove_LI(path)
