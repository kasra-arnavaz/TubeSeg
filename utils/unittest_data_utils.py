from data_utils import *
import unittest

class TestDataUtils(unittest.TestCase):

## NameParser
    def test_name_parser_unpacking(self):
        n = NameParser('pred-0.7-semi-40_tr_LI-2018-09-28-emb5-pos1_tp53-A1_.tif')
        self.assertEqual(n.category, 'pred-0.7-semi-40')
        self.assertEqual(n.split, 'tr')
        self.assertEqual(n.movie_name, 'LI-2018-09-28-emb5-pos1')
        self.assertEqual(n.time_point, 'tp53-A1')
        self.assertEqual(n.patch_id, '')
        self.assertEqual(n.extension, '.tif')
    
    def test_name_parser_packing(self):
        n = NameParser('duct_tr_LI-2018-09-28-emb5-pos1_tp53-A1_.tif')
        n.patch_id = 'D4'
        self.assertEqual(n.name, 'duct_tr_LI-2018-09-28-emb5-pos1_tp53-A1_D4.tif')

    def test_frame_name(self):
        n = NameParser('duct_tr_LI-2018-09-28-emb5-pos1_tp53-A1_B4.tif')
        self.assertEqual(n.frame_name, 'LI-2018-09-28-emb5-pos1_tp53-A1')

## TifReader
    def test_tif_reader_read(self):
        _, x = TifReader('utils/unittest_data/patches_from_images/patches')\
            .read('label_val_LI-2016-03-04-emb5-pos2_tp105-A1D3D4_B3.tif')
        self.assertEqual(x.shape, (27,256,256))
    
    def test_tif_reader_names(self):
        names = TifReader('utils/unittest_data/patches_from_images/patches').names
        self.assertEqual(names, ['label_val_LI-2016-03-04-emb5-pos2_tp105-A1D3D4_B3.tif',\
                                'label_val_LI-2018-12-07-emb6-pos4_tp41-A3_B3.tif'])

    def test_tif_reader_frame_names(self):
        names = TifReader('utils/unittest_data/patches_from_images/patches').frame_names
        self.assertEqual(names, ['LI-2016-03-04-emb5-pos2_tp105-A1D3D4',\
                                 'LI-2018-12-07-emb6-pos4_tp41-A3'])
    
    def test_tif_reader_next(self):
        names, xs = [], []
        for name, x in TifReader('utils/unittest_data/patches_from_images/patches'):
            names.append(name.name)
            xs.append(x)
        self.assertEqual(names, ['label_val_LI-2016-03-04-emb5-pos2_tp105-A1D3D4_B3.tif',\
                                'label_val_LI-2018-12-07-emb6-pos4_tp41-A3_B3.tif'])
        self.assertEqual(xs[0].shape, (27,256,256))
        self.assertEqual(xs[1].shape, (33,256,256))

## TifWriter
    def test_tif_writer_write(self):
        x = np.ones([10,10])
        n = NameParser('ones_val_LI-2018-12-07-emb6-pos4_tp41-A3_B3.tif')
        TifWriter('utils/unittest_data', n, x).write()
        _, y = TifReader('utils/unittest_data').read('ones_val_LI-2018-12-07-emb6-pos4_tp41-A3_B3.tif')
        self.assertEqual(y.sum(), 100)


unittest.main()
