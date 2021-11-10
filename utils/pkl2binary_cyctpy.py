import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from typing import List, Tuple

from tifffile.tifffile import imagej_shape
from utils.unpickle import read_pickle
from graph.nx_graph import Cycle, Component, NxGraph
from time_filtering.trackpy_filtering import TrackpyFiltering

class Topo2Tif:

    def __init__(self, topology_path, topology_name, pred_file):
        self.topology_path = topology_path
        self.topology_name = topology_name
        self.pred_file = pred_file
        self.names = [name for name in os.listdir(self.topology_path) if name.endswith('.cyctpy')]

    def topology(self, name):
        return read_pickle(f'{self.topology_path}/{name}')
    
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
        
        colors = 255*np.array(cm.Set1.colors)
        colors = colors.astype('uint8')
        # with tif.TiffWriter('temp.tif') as tiff:
        #     for tp in range(5):
        #         binary = np.zeros(self.shape + (3,), dtype='uint8') # 3 for rgb channels
        #         topology = self.topology(f'{self.topology_name}_tp{tp+1}.cyctpy')
        #         for edges, loop_id in zip(topology.topology_edges, topology.loop_id):
        #             binary[self.skel_indices(edges, topology)] = colors[loop_id%len(colors)]
        #         tiff.save(binary, contiguous=True)
        binary = np.zeros((5,)+self.shape+(3,), dtype='uint8')
        for tp in range(5):
            topology = self.topology(f'{self.topology_name}_tp{tp+1}.cyctpy')
            for edges, loop_id in zip(topology.topology_edges, topology.loop_id):
                    binary[tp][self.skel_indices(edges, topology)] = colors[loop_id%len(colors)]
        tif.imwrite('qwer.tif', binary, imagej, metadata={'axes': 'TZYXC'})


    def write_npy(self, saving_path=None):
        if saving_path is None: saving_path = self.path
        np.save(f'{saving_path}/{self.name}.npy', self.binary())
    
    def show_mip(self):
        mip = np.amax(self.binary(), axis=0)
        plt.imshow(mip)
        plt.show()




    
if __name__ == '__main__':
    t2t = Topo2Tif('movie/test/LI_2019-07-03_emb7_pos2/cyc/srch=15, mem=1, thr=15, step=0.9, stop=5', 'pred-0.7-semi-40_2019-07-03_emb7_pos2',
    'movie/test/LI_2019-07-03_emb7_pos2/pred/pred-0.7-semi-40_2019-07-03_emb7_pos2_tp38.tif')
    t2t.write_tif()
    # t2t.show_mip()
    
