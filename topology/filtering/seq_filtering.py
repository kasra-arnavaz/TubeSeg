from scipy.spatial.distance import cdist
import pickle
import networkx as nx
from cached_property import cached_property
from argparse import ArgumentParser
import os

from topology.cyc_cmp import Topology, Component, Cycle, NxGraph
from topology.unpickle import read_pickle


class SequentialFiltering:
    ''' Components which are peresent for only one timepoint are filtered out
    '''

    def __init__(self, cmp_path: str, cmp_name: str, thr: float, write_path: str = None):
        if write_path is None: write_path = cmp_path
        self.cmp_path = cmp_path
        self.cmp_name = cmp_name
        self.thr = thr
        self.write_path = write_path
        self.num_tp = len([file for file in os.listdir(cmp_path) if file.endswith('cmp')])

    @cached_property
    def all_topology(self):
        return [read_pickle(f'{self.cmp_path}/{self.cmp_name}_tp{i+1}.cmp') for i in range(self.num_tp)]
        
    @staticmethod
    def min_pointclould_distance(pointcloud1, pointcloud2):
        # Pointclouds in N x 3 format
        D = cdist(pointcloud1, pointcloud2, metric='euclidean')
        return min(D.flatten())

    @cached_property
    def time_filter(self):
        topo = self.all_topology
        keep_idx = {}
        for i in range(self.num_tp):
            if i == 0: keep_idx[i] = self.comparative_filter(topo[i], topo[i+1])
            elif i == self.num_tp-1: keep_idx[i] = self.comparative_filter(topo[i], topo[i-1])
            else:
                keep_prev = self.comparative_filter(topo[i], topo[i-1])
                keep_next = self.comparative_filter(topo[i], topo[i+1])
                keep_idx[i] = list(set(keep_prev) | set(keep_next))
        return keep_idx

    def comparative_filter(self, topo_i, topo_j):
        keep_idx = []
        for i, pos_i in topo_i.position.items():
            for j, pos_j in topo_j.position.items():
                if self.min_pointclould_distance(pos_i, pos_j) < self.thr:
                    keep_idx.append(i)
        return list(set(keep_idx))

    def save_filtered(self):
        for tp in range(self.num_tp):
            self.write_pickle(tp)
        
    def write_pickle(self, tp):
        if f'{self.cmp_name}_tp{tp+1}.{self.topo_type}' not in os.listdir(self.write_path):
            with open(f'{self.write_path}/{self.cmp_name}_tp{tp+1}.{self.topo_type}', 'wb') as f:
                pickle.dump(self.set_all_properties(tp), f)
        elif os.path.getsize(f'{self.write_path}/{self.cmp_name}_tp{tp+1}.{self.topo_type}') < 1000:
            with open(f'{self.write_path}/{self.cmp_name}_tp{tp+1}.{self.topo_type}', 'wb') as f:
                pickle.dump(self.set_all_properties(tp), f)

    def set_all_properties(self, tp):
        topo = self.all_topology[tp]
        keep_idx = self.time_filter[tp]
        self.position = {key:value for key, value in topo.position.items() if key in keep_idx}
        self.topology_nodes = [list for i, list in enumerate(topo.topology_nodes) if i in keep_idx]
        self.topology_edges = [edges for i, edges in enumerate(topo.topology_edges) if i in keep_idx]
        self.center = {key:value for key, value in topo.center.items() if key in keep_idx}
        self.center_xy = {key:value for key, value in topo.center_xy.items() if key in keep_idx}
        self.node_list = nx.utils.misc.flatten(self.topology_nodes)
        self.edge_list = [edge for edge in topo.edge_list if (edge[0] in self.node_list) and (edge[1] in self.node_list)]
        self.nodes_xy_position = topo.nodes_xy_position
        return self

    @property
    def topo_type(self):
        return f'cmpseq{self.thr}'



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cmp_path', type=str)
    parser.add_argument('--cmp_name', type=str)
    parser.add_argument('--thr', type=float, default=15)
    args = parser.parse_args()
    SequentialFiltering(args.cmp_path, args.cmp_name, args.thr).save_filtered()
