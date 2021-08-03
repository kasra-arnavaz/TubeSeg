from utils.data_utils import *
from abc import ABC, abstractmethod

class Conversion(ABC):

    def __init__(self, name_parser: NameParser, input_pic: np.ndarray):
        self.name_parser = name_parser
        self.input_pic = input_pic
    
    @abstractmethod
    def convert(self, *args):
        pass


class Duct2Mip(Conversion):

    def convert(self, *args):
        mip = np.amax(self.input_pic, axis=0)   # maximum intensity projection
        self.name_parser.category = 'duct-mip'
        return self.name_parser, mip


class Prob2Pred(Conversion):

    def convert(self, *args):
        threshold = args[0]
        pred = ((self.input_pic >= threshold)*255).astype('uint8')
        self.name_parser.category = self.name_parser.category.replace('prob', f'pred-{threshold}')
        return self.name_parser, pred


class Prob2Ent(Conversion):

    def convert(self, *args):
        from scipy.stats import entropy
        p = np.stack((self.input_pic, 1-self.input_pic), axis=0)
        ent = entropy(p, base=2, axis=0)
        self.name_parser.category = self.name_parser.category.replace('prob', 'ent')
        return self.name_parser, ent
    

class Lin2Prob(Conversion):

    def convert(self, *args):
        a, b = args
        z = a*self.input_pic + b
        prob = 1/(1+np.exp(-z))
        self.name_parser.category = self.name_parser.category.replace('lin', 'prob')
        return self.name_parser, prob


class Ent2ThrEnt(Conversion):

    def convert(self, *args):
        threshold = args[0]
        thr_ent = ((self.input_pic >= threshold)*255).astype('uint8')
        self.name_parser.category = self.name_parser.category.replace('ent', f'ent-{threshold}')
        return self.name_parser, thr_ent


class MassDataConversion:

    def __init__(self, read_path, write_path, conversion_type: Conversion, *args):
        self.read_path = read_path
        self.conversion_type = conversion_type
        self.write_path = write_path
        self.args = args
    
    def read_convert_write(self):
        for input_name, input_pic in TifReader(self.read_path):
            output_name, output_pic = self.conversion_type(input_name, input_pic).convert(*self.args)
            TifWriter(self.write_path, output_name, output_pic).write()


if __name__ == '__main__':
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        MassDataConversion('results/unetcldice/2d/val', f'results/unetcldice/2d/val/pred-{i}', Prob2Pred, i).read_convert_write()
