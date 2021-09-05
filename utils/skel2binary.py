import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt

class Skeletonize:

    def __init__(self, path, name):
        self.path = path
        self.name = name
    
    @property
    def shape(self):
        return tif.imread(f'{self.path}/{self.name}.tif').shape

    def read_lines(self):
        with open(f'{self.path}/skel_{self.name}.graph', 'r') as f:
            lines = f.readlines()
        return lines

    def read_nodes(self):
        node_pos_list = [list(line[2:-1].split(' ')) for line in self.read_lines() if line.startswith('n')]
        node_pos_arr = np.array(node_pos_list, dtype=float)
        node_pos_arr[:,0] /= 2.89
        return np.round(node_pos_arr).astype(int)

    def read_edges(self):
        edge_list = [list(line[2:-1].split(' ')) for line in self.read_lines() if line.startswith('c')]
        return np.array(edge_list, dtype=int)

    @staticmethod
    def line_segment(p, q):
        return set([tuple(np.round(t*p + (1-t)*q).astype(int)) for t in np.linspace(0,1,10)])
    
    def skel_indices(self):
        idx = set()
        nodes = self.read_nodes()
        for i, j in self.read_edges():
            idx |= self.line_segment(nodes[i], nodes[j])
        z_list, y_list, x_list = [], [], []
        for z, y, x in list(idx):
            z_list.append(z)
            y_list.append(y)
            x_list.append(x)
        return tuple(z_list), tuple(y_list), tuple(x_list)
    
    def binary(self):
        binary = np.zeros(self.shape)
        binary[self.skel_indices()] = 1
        return binary

    def write_npy(self, saving_path=None):
        if saving_path is None: saving_path = self.path
        np.save(f'{saving_path}/{self.name}.npy', self.binary())
    
    def show_mip(self):
        mip = np.amax(self.binary(), axis=0)
        plt.imshow(mip)
        plt.show()




    
    
        
if __name__ == '__main__':
    s2b = Skeletonize('D:/dataset/test/patches/label', 'label_ts_LI-2019-01-17-emb7-pos4_tp158-C1D4_D3')
    s2b.show_mip()
