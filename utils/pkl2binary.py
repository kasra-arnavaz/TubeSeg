import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from typing import List, Tuple
from utils.unpickle import read_pickle
from graph.nx_graph import Cycle, Component, NxGraph
from matplotlib import cm

class Topo2Tif:

    def __init__(self, topology_file, pred_file, saved_file):
        self.topology_file = topology_file
        self.pred_file = pred_file
        self.saved_file = saved_file

    def topology(self):
        return read_pickle(self.topology_file)
    
    @property
    def shape(self):
        return tif.imread(self.pred_file).shape

    def all_edge_connections(self):
        return self.topology().topology_edges

    def nodes_position(self):
        list_node_pos = [v for v_list in self.topology().position.values() for v in v_list]
        pos_dict = {}
        for node, pos in zip(self.topology().node_list, list_node_pos):
            pos[0] /= 2.89
            pos = np.round(pos).astype(int)
            pos_dict[node] = pos
        return pos_dict

    @staticmethod
    def line_segment(p, q):
        return set([tuple(np.round(t*p + (1-t)*q).astype(int)) for t in np.linspace(0,1,100)])
    
    def skel_indices(self, edge_connectivity: List[Tuple[int]]):
        idx = set()
        nodes_pos = self.nodes_position()
        for i, j in edge_connectivity:
            idx |= self.line_segment(nodes_pos[i], nodes_pos[j])
        z_list, y_list, x_list = [], [], []
        for z, y, x in list(idx):
            z_list.append(z)
            y_list.append(y)
            x_list.append(x)
        return tuple(z_list), tuple(y_list), tuple(x_list)
    
    def binary(self):
        binary = np.zeros(self.shape + (3,), dtype='uint8') # 3 for rgb channels
        colors = 255*np.array(cm.Set1.colors)
        colors = colors.astype('uint8')
        for num, i in enumerate(self.all_edge_connections()):
            if num % 5 == 0: binary[self.skel_indices(i)] = [255, 255, 255]
            else: binary[self.skel_indices(i)] = colors[num%9]

            # binary[self.skel_indices(i)] = np.random.randint(256, size=3)

        return binary

    def write_tif(self):
        tif.imwrite(self.saved_file, self.binary(), photometric='rgb')

    def write_npy(self, saving_path=None):
        if saving_path is None: saving_path = self.path
        np.save(f'{saving_path}/{self.name}.npy', self.binary())
    
    def show_mip(self):
        mip = np.amax(self.binary(), axis=0)
        plt.imshow(mip)
        plt.show()




    
if __name__ == '__main__':
    t2t = Topo2Tif('movie/test/LI_2019-07-03_emb7_pos2/cyc/pred-0.7-semi-40_2019-07-03_emb7_pos2_tp38.cyc',
    'movie/test/LI_2019-07-03_emb7_pos2/pred/pred-0.7-semi-40_2019-07-03_emb7_pos2_tp38.tif', 'movie/test/LI_2019-07-03_emb7_pos2/cyc/bincyc_pred-0.7-semi-40_2019-07-03_emb7_pos2_tp38.tif')
    # t2t.write_tif()
    t2t.show_mip()
    
