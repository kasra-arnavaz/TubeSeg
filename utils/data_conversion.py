import tifffile as tif
import numpy as np
import os
from abc import ABC, abstractmethod, abstractproperty

class Conversion(ABC):

    def __init__(self, input_name: str, input_img: np.ndarray, write_path:str, *args):
        self.input_name = input_name
        self.input_img = input_img
        self.write_path = write_path
        self.args = args
    
    @abstractmethod
    def convert(self):
        pass

    @abstractproperty
    def output_name(self):
        pass

    def write(self):
        os.makedirs(self.write_path, exist_ok=True)
        tif.imwrite(f'{self.write_path}/{self.output_name}', self.convert())




class Duct2Mip(Conversion):
    # maximum intensity projection
    def convert(self):
        return np.amax(self.input_img, axis=0)

    @property
    def output_name(self):
        return self.input_name.replace('duct', 'duct-mip')


class Prob2Pred(Conversion):

    def convert(self):
        return ((self.input_img >= self.args[0])*255).astype('uint8')
    
    @property
    def output_name(self):
        return self.input_name.replace('prob', f'pred-{self.args[0]}')


class Prob2Ent(Conversion):

    def convert(self):
        from scipy.stats import entropy
        p = np.stack((self.input_img, 1-self.input_img), axis=0)
        return entropy(p, base=2, axis=0)
    
    @property
    def output_name(self):
        return self.input_name.replace('prob', 'ent')
    

class Lin2Prob(Conversion):

    def convert(self):
        a, b = self.args
        z = a*self.input_img + b
        return 1/(1+np.exp(-z))

    @property
    def output_name(self):
        return self.input_name.replace('lin', 'prob')



def MassDataConversion(input_path:str , write_path: str, converter: Conversion, *args):
    names = [name for name in os.listdir(input_path) if name.endswith('.tif')]
    for name in names:
        input_img = tif.imread(f'{input_path}/{name}')
        converter(name, input_img, write_path, args).write()


if __name__ == '__main__':
    MassDataConversion('../alaki', '../alaki/mip', Duct2Mip)
