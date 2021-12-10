import unittest
from model_utils import *
from model_utils_new import *

class TestModelUtils(unittest.TestCase):

    def test_extract_and_transform_training_images(self):
        old = extract_and_transform_training_images()