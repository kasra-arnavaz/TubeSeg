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

    def __init__(self, cyc_path, pred_file, cyctpy_path=None):
        if cyctpy_path is None: cyctpy_path = cyc_path
        self.cyc_path = cyc_path
        self.cyctpy_path = cyctpy_path
        self.pred_file = pred_file
        self.names = [name.replace('.cyctpy', '') for name in os.listdir(self.cyctpy_path) if name.endswith('.cyctpy')]

    def topology(self, path, name, extension):
        return read_pickle(f'{path}/{name}.{extension}')
    
    @property
    def shape(self):
        return tif.imread(self.pred_file).shape

    def nodes_position(self, topology):
        list_node_pos = [v for v_list in topology.position.values() for v in v_list]
        pos_dict = {}
        for node, pos in zip(topology.node_list, list_node_pos):
            pos[0] /= 2.89
            pos = np.round(pos).astype(int)
            pos_dict[node] = pos
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
        num_frames = len(self.names)
        # with tif.TiffWriter('temp.tif') as tiff:
        #     for tp in range(5):
        #         binary = np.zeros(self.shape + (3,), dtype='uint8') # 3 for rgb channels
        #         topology = self.topology(f'{self.topology_name}_tp{tp+1}.cyctpy')
        #         for edges, loop_id in zip(topology.topology_edges, topology.loop_id):
        #             binary[self.skel_indices(edges, topology)] = colors[loop_id%len(colors)]
        #         tiff.save(binary, contiguous=True)
        binary = np.zeros((num_frames,)+self.shape+(3,), dtype='uint8')
        for tp, name in enumerate(self.names):
            print(name)
            cyc = self.topology(self.cyc_path, name, 'cyc')
            cyctpy = self.topology(self.cyctpy_path, name, 'cyctpy')
            i = 0
            for cyc_edges in cyc.topology_edges:
                if not cyc_edges in cyctpy.topology_edges: #filtered
                    binary[tp][self.skel_indices(cyc_edges, cyc)] = [np.random.randint(256), 0, 0]
                else:
                    np.random.seed(cyctpy.loop_id[i])
                    binary[tp][self.skel_indices(cyc_edges, cyc)] = [0, np.random.randint(256), np.random.randint(256)]
                    i += 1
        tif.imwrite('mock.tif', binary, imagej=True, metadata={'axes': 'TZYXC'})


    def write_npy(self, saving_path=None):
        if saving_path is None: saving_path = self.path
        np.save(f'{saving_path}/{self.name}.npy', self.binary())
    
    def show_mip(self):
        mip = np.amax(self.binary(), axis=0)
        plt.imshow(mip)
        plt.show()




if __name__ == '__main__':
    t2t = Topo2Tif('mock_movie', 'movie/test/LI_2018-11-20_emb7_pos4/pred/pred-0.7-semi-40_2018-11-20_emb7_pos4_tp38.tif')
    t2t.write_tif()

    # t2t.show_mip()
