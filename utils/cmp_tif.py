import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from typing import List, Tuple

from utils.unpickle import read_pickle
from graph.nx_graph import Cycle, Component, NxGraph
from time_filtering.trackpy_filtering import TrackpyFiltering

class Topo2Tif:

    def __init__(self, cmp_path, cmp_name, pred_file):
        self.cmp_path = cmp_path
        self.pred_file = pred_file
        self.num_frames = len([name for name in os.listdir(self.cmp_path) if name.endswith('.cmp')])
        #self.num_frames = 3
        self.names = [f'{cmp_name}_tp{i+1}' for i in range(self.num_frames)]

    def topology(self, name, extension):
        return read_pickle(f'{self.cmp_path}/{name}.{extension}')
    
    @property
    def shape(self):
        return tif.imread(self.pred_file).shape

    def nodes_position(self, topology):
        list_node_pos = [v for v_list in topology.position.values() for v in v_list]
        pos_dict = {}
        for node, pos in zip(topology.node_list, list_node_pos):
            position = pos / (np.array([2.89, 1, 1]))
            pos_dict[node] = np.round(position).astype(int)
        return pos_dict

    @staticmethod
    def line_segment(p, q):
        return set([tuple(np.round(t*p + (1-t)*q).astype(int)) for t in np.linspace(0,1,100)])
    
    def skel_indices(self, edge_connectivity: List[Tuple[int]], topology):
        idx = set()
        nodes_pos = self.nodes_position(topology)
        for i, j in edge_connectivity:
            idx |= self.line_segment(nodes_pos[i], nodes_pos[j])
        z_list, y_list, x_list = [], [], []
        for z, y, x in list(idx):
            z_list.append(z)
            y_list.append(y)
            x_list.append(x)
        return tuple(z_list), tuple(y_list), tuple(x_list)
    
    def write_tif(self):

        binary = np.zeros((self.num_frames,)+self.shape+(3,), dtype='uint8')
        for tp, name in enumerate(self.names):
            print(name)
            cmp = self.topology(name, 'cmp')
            cmpseq = self.topology(name, 'cmpseq5')
            i = 0
            for cmp_edges in cmp.topology_edges:
                if not cmp_edges in cmpseq.topology_edges: #filtered
                    binary[tp][self.skel_indices(cmp_edges, cmp)] = [np.random.randint(256), 0, 0]
                else:
                    binary[tp][self.skel_indices(cmp_edges, cmp)] = [0, np.random.randint(256), np.random.randint(256)]
                    i += 1
        tif.imwrite('mock_cmp.tif', binary, imagej=True, metadata={'axes': 'TZYXC'})


    def write_npy(self, saving_path=None):
        if saving_path is None: saving_path = self.path
        np.save(f'{saving_path}/{self.name}.npy', self.binary())
    
    def show_mip(self):
        mip = np.amax(self.binary(), axis=0)
        plt.imshow(mip)
        plt.show()




if __name__ == '__main__':
    t2t = Topo2Tif('mock_movie', 'pred-0.7-semi-40_2018-11-20_emb7_pos4', '../results/test-old/LI_2018-11-20_emb7_pos4/pred/pred-0.7-semi-40_2018-11-20_emb7_pos4_tp38.tif')
    t2t.write_tif()

    # t2t.show_mip()
