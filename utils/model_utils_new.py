from utils.data_utils import *
from utils.preprocess_data import *
from abc import ABC, abstractmethod

class PrepareData(ABC):

    @abstractmethod
    def extract_images(self):
        pass
    
    @abstractmethod
    def transform_images(self):
        pass

    @abstractmethod
    def load_patches(self):
        pass

class PrepareTestData(PrepareData):

    def __init__(self, x, transformer: TransformData):
        self.x = x
        self.tranformer = transformer

    def extract_data(self):
        pass

    def transform_data(self):
        return self.transformer(self.x).transform()

    def load_data(self):
        ## to do: fix self.output_size, input_size, grid_size
        x_transformed = self.transform_data(self.x)
        x_pad = np.pad(x_transformed, ((0,0),(self.margin,self.margin),(self.margin,self.margin)), 'constant', constant_values=0)
        x_pad = np.expand_dims(x_pad, axis=-1)
        for z in range(x_transformed.shape[0]):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    yield x_pad[np.newaxis ,z, i*self.output_size:i*self.output_size+self.input_size,\
                                                j*self.output_size:j*self.output_size+self.input_size,:]

class PrepareTrainDataUNet(PrepareData):

    def __init__(self, x_path, y_path, transformer: TransformData):
        self.x_path = x_path
        self.y_path = y_path

    def extract_data(self):
        x_list, y_list = [], []
        for x_name, x in TifReader(self.x_path):
            y_name = x_name.copy()
            y_name.category = 'label'
            y = TifReader(self.y_path).read(y_name.name)
            x_list.append(x)
            y_list.append(y)
        return np.array(x_list), np.array(y_list)

    def transform_data(self):
        return self.trans            
