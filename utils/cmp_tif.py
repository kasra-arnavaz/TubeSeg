import numpy as np
from numpy.lib.npyio import save
import tifffile as tif
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
from argparse import ArgumentParser

from utils.unpickle import read_pickle
from graph.nx_graph import Cycle, Component, NxGraph    
from time_filtering.trackpy_filtering import TrackpyFiltering
from time_filtering.seq_filtering import SequentialFiltering

class Topo2Tif:

    def __init__(self, cmp_path, pred_prefix):
        self.cmp_path = cmp_path
        self.pred_prefix = pred_prefix
        self.num_frames = len([name for name in os.listdir(self.cmp_path) if name.endswith('.cmp')])
        self.movie_name = cmp_path.split('/')[-2].replace('LI_', '')
        self.names = [f'{pred_prefix}_{self.movie_name}_tp{i+1}' for i in range(self.num_frames)]

    def topology(self, name, extension):
        return read_pickle(f'{self.cmp_path}/{name}.{extension}')
    
    @property
    def shape(self):
        return tif.imread(f'{self.cmp_path}/../pred/{self.names[0]}.tif').shape

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
    
    def write_tif(self, filtered_cmp_ext, save_mip=True):
        saved_name = f'{filtered_cmp_ext}-{self.pred_prefix}_{self.movie_name}'
        binary = np.zeros((self.num_frames,)+self.shape+(3,), dtype='uint8')
        if f'{saved_name}.tif' not in os.listdir(f'{self.cmp_path}/..'):
           for tp, name in enumerate(self.names):
               cmp = self.topology(name, 'cmp')
               cmpseq = self.topology(name, filtered_cmp_ext)
               for cmp_edges in cmp.topology_edges:
                   if not cmp_edges in cmpseq.topology_edges: #filtered
                       binary[tp][self.skel_indices(cmp_edges, cmp)] = [np.random.randint(50,256), 0, 0]
                   else:
                       binary[tp][self.skel_indices(cmp_edges, cmp)] = [0, np.random.randint(50,256), np.random.randint(50,256)]
           tif.imwrite(f'{self.cmp_path}/../{saved_name}.tif', binary, imagej=True, metadata={'axes': 'TZYXC'})
           if save_mip:
               tif.imwrite(f'{self.cmp_path}/../mip{saved_name}.tif', np.amax(binary, axis=1))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cmppath', type=str)
    args = parser.parse_args()
    pred_prefix = 'pred-0.7-semi-40'
    Topo2Tif(args.cmppath, pred_prefix).write_tif(filtered_cmp_ext='cmpseq3', save_mip=False)
