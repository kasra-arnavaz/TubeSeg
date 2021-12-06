import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from typing import List, Tuple
from argparse import ArgumentParser

from utils.unpickle import read_pickle
from graph.nx_graph import Cycle, Component, NxGraph
from time_filtering.trackpy_filtering import TrackpyFiltering

class Topo2Tif:

    def __init__(self, cyc_path, pred_prefix):
        self.cyc_path = cyc_path
        self.pred_prefix = pred_prefix
        self.num_frames = len([name for name in os.listdir(self.cyc_path) if name.endswith('.cyc')])
        self.movie_name = cyc_path.split('/')[-2].replace('LI_', '')
        self.names = [f'{pred_prefix}_{self.movie_name}_tp{i+1}' for i in range(self.num_frames)]

    def topology(self, name, extension):
        return read_pickle(f'{self.cyc_path}/{name}.{extension}')
    
    @property
    def shape(self):
        return tif.imread(f'{self.cyc_path}/../pred/{self.names[0]}.tif').shape

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
    
    def write_tif(self, filtered_cyc_ext, save_mip=True):
        # with tif.TiffWriter('temp.tif') as tiff:
        #     for tp in range(5):
        #         binary = np.zeros(self.shape + (3,), dtype='uint8') # 3 for rgb channels
        #         topology = self.topology(f'{self.topology_name}_tp{tp+1}.cyctpy')
        #         for edges, loop_id in zip(topology.topology_edges, topology.loop_id):
        #             binary[self.skel_indices(edges, topology)] = colors[loop_id%len(colors)]
        #         tiff.save(binary, contiguous=True)
        saved_name = f'{filtered_cyc_ext}-{self.pred_prefix}_{self.movie_name}'
        binary = np.zeros((self.num_frames,)+self.shape+(3,), dtype='uint8')
        if f'{saved_name}.tif' not in os.listdir(f'{self.cyc_path}/..'):
           print(saved_name)
           for tp, name in enumerate(self.names):
               cyc = self.topology(name, 'cyc')
               cyctpy = self.topology(name, filtered_cyc_ext)
               i = 0
               for cyc_edges in cyc.topology_edges:
                   if not cyc_edges in cyctpy.topology_edges: #filtered
                      binary[tp][self.skel_indices(cyc_edges, cyc)] = [np.random.randint(50,256), 0, 0]
                   else:
                       np.random.seed(cyctpy.loop_id[i])
                       binary[tp][self.skel_indices(cyc_edges, cyc)] = [0, np.random.randint(50,256), np.random.randint(50,256)]
                       i += 1
           tif.imwrite(f'{self.cyc_path}/../{saved_name}.tif', binary, imagej=True, metadata={'axes': 'TZYXC'})
           if save_mip:
               tif.imwrite(f'{self.cyc_path}/../mip{saved_name}.tif', np.amax(binary, axis=1))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cycpath', type=str)
    args = parser.parse_args()
    pred_prefix = 'pred-0.7-semi-40'
    Topo2Tif(args.cycpath, pred_prefix).write_tif(filtered_cyc_ext='cyctpy15', save_mip=False)
