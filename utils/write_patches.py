from data_conversion import *

class PatchSelection:
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


class WriteMultiplePatches:
    
    def __init__(self, image_path, patch_path, write_path, patch_selector=PatchSelection):
        self.image_path = image_path
        self.patch_path = patch_path
        self.write_path = write_path
        self.patch_selector = patch_selector

    def patches_from_images(self):
        image_frame_names = TifReader(self.image_path).frame_names
        patch_frame_names = TifReader(self.patch_path).frame_names
        return set(patch_frame_names) == set(image_frame_names) 
    
    def read_select_write(self):
        if self.patches_from_images:
            for patch_name, _ in TifReader(self.patch_path):
                for image_name, image in TifReader(self.image_path):
                    if patch_name.frame_name == image_name.frame_name:
                        write_name, write_patch = self.patch_selector(image_name, image).select_patch(patch_name.patch_id)
                        TifWriter(self.write_path, write_name, write_patch).write()
        else:
            raise ValueError(f"Patches in {self.patch_path} don't come from images in {self.image_path}!")

if __name__ == '__main__':
    WriteMultiplePatches('E:/Dataset/test/images/ducts', 'E:/Dataset/test/patches/labels','E:/Dataset/test/patches/ducts').read_select_write()
        
