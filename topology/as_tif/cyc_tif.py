import numpy as np
import tifffile as tif
import os
from typing import List, Tuple
from argparse import ArgumentParser

from topology.unpickle import read_pickle
from topology.cyc_cmp import Cycle, Component, NxGraph
from topology.filtering.tpy_filtering import TrackpyFiltering

class CycTif:

    def __init__(self, cyc_path: str, cyc_name: str, filtered_cyc_ext: str, pred_path: str = None,\
        write_path: str = None, make_mip: bool = False) -> None:
        if write_path is None: write_path = cyc_path
        if pred_path is None: pred_path = f'{cyc_path}/../pred'
        self.cyc_path = cyc_path
        self.pred_path = pred_path
        self.write_path = write_path
        self.make_mip = make_mip
        self.filtered_cyc_ext = filtered_cyc_ext
        self.cyc_name = cyc_name
        self.num_tp = len([name for name in os.listdir(self.cyc_path) if name.endswith('.cyc')])
        self.names = [f'{cyc_name}_tp{i+1}' for i in range(self.num_tp)]

    def topology(self, name, extension):
        return read_pickle(f'{self.cyc_path}/{name}.{extension}')
    
    @property
    def shape(self):
        return tif.imread(f'{self.pred_path}/{self.names[0]}.tif').shape

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
        saved_name = f'{self.filtered_cyc_ext}-{self.cyc_name}'
        binary = np.zeros((self.num_tp,)+self.shape+(3,), dtype='uint8')
        if f'{saved_name}.tif' not in os.listdir(self.write_path):
           for tp, name in enumerate(self.names):
               cyc = self.topology(name, 'cyc')
               cyctpy = self.topology(name, self.filtered_cyc_ext)
               i = 0
               for cyc_edges in cyc.topology_edges:
                   if not cyc_edges in cyctpy.topology_edges: #filtered
                      binary[tp][self.skel_indices(cyc_edges, cyc)] = [np.random.randint(50,256), 0, 0]
                   else:
                       np.random.seed(cyctpy.loop_id[i])
                       binary[tp][self.skel_indices(cyc_edges, cyc)] = [0, np.random.randint(50,256), np.random.randint(50,256)]
                       i += 1
           tif.imwrite(f'{self.write_path}/{saved_name}.tif', binary, imagej=True, metadata={'axes': 'TZYXC'})
           if self.make_mip:
               tif.imwrite(f'{self.write_path}/mip-{saved_name}.tif', np.amax(binary, axis=1))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cyc_path', type=str)
    parser.add_argument('--cyc_name', type=str)
    parser.add_argument('--filtered_cyc_ext', type=str)
    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--write_path', type=str)
    parser.add_argument('--make_mip', action='store_true', default=False)
    args = parser.parse_args()
    CycTif(args.cyc_path, args.cyc_name, args.filtered_cyc_ext, args.pred_path, args.write_path, args.make_mip).write_tif()
