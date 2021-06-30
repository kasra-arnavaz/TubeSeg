from write_patches import *
import unittest

class TestWritePatches(unittest.TestCase):
    
    def test_select_patch(self):
        name, image = TifReader('utils/unittest_data/patches_from_images/images')\
                                .read('duct_val_LI-2016-03-04-emb5-pos2_tp105-A1D3D4_.tif')
        patch_name, patch = PatchSelection(name, image).select_patch('A4')
        self.assertEqual(patch.sum(), image[:,:256,:256].sum())
        self.assertEqual(patch_name.name, 'duct_val_LI-2016-03-04-emb5-pos2_tp105-A1D3D4_A4.tif')

    def test_patches_not_from_images(self):
        path = 'utils/unittest_data/patches_not_from_images'
        wmp = WriteMultiplePatches(f'{path}/images', f'{path}/patches', '')
        self.assertFalse(wmp.patches_from_images())

    def test_patches_from_images(self):
        path = 'utils/unittest_data/patches_from_images'
        wmp = WriteMultiplePatches(f'{path}/images', f'{path}/patches', '')
        self.assertTrue(wmp.patches_from_images())

    def test_read_select_write(self):
        '''Manual visual check'''
        WriteMultiplePatches('utils/unittest_data/patches_from_images/images', 'utils/unittest_data/patches_from_images/patches',\
                        'utils/unittest_data/patches_from_images/written').read_select_write()

unittest.main()


