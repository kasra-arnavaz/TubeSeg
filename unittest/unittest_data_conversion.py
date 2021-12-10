from data_conversion import *
import unittest

class TestDataConversion(unittest.TestCase):
    '''Manual visual check'''

    def test_duct2mip(self):
        MassDataConversion('utils/unittest_data/duct', 'utils/unittest_data/mip', Duct2Mip).read_convert_write()

    def test_lin2prob(self):
        MassDataConversion('utils/unittest_data/lin', 'utils/unittest_data/prob', Lin2Prob, 1, 0).read_convert_write()

    def test_prob2ent(self):
        MassDataConversion('utils/unittest_data/prob', 'utils/unittest_data/ent', Prob2Ent).read_convert_write()

    def test_prob2pred(self):
        MassDataConversion('utils/unittest_data/prob', 'utils/unittest_data/pred', Prob2Pred, 0.5).read_convert_write()

    def test_ent2thr_ent(self):
        MassDataConversion('utils/unittest_data/ent', 'utils/unittest_data/ent/ent-0.7', Ent2ThrEnt, 0.7).read_convert_write()

unittest.main()
