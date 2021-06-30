import numpy as np
from abc import ABC, abstractmethod

class TransformData(ABC):

    def __init__(self, x):
        self.x = x

    @abstractmethod
    def transform(self):
        pass

class ModifiedStandardization(TransformData):

    @property
    def mean(self):
        return np.mean(self.x, axis=(1,2)).reshape(-1,1,1)
    
    @property
    def std(self):
        return np.std(self.x, axis=(1,2)).reshape(-1,1,1)

    def transform(self):
        # Set intensities farther than 3*std to zero.
        z = (self.x - self.mean) / (3*self.std)
        return np.clip(z, 0, 1)
        