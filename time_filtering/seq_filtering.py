from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
import pickle
import networkx as nx
from cached_property import cached_property

from graph.nx_graph import Topology, Component, Cycle, NxGraph
from utils.unpickle import read_pickle


class SequentialFiltering:

    def __init__(self, path, name, extension, tp_max ,threshold):
        self.path = path
        self.name = name
        self.extension = extension
        self.tp_max = tp_max
        self.threshold = threshold

    @cached_property
    def all_topology(self):
        return [read_pickle(f'{self.path}/{self.name}_tp{i+1}.{self.extension}') for i in range(self.tp_max)]

    @staticmethod
    def min_pointclould_distance(pointcloud1, pointcloud2):
        # Pointclouds in N x 3 format
        D = cdist(pointcloud1, pointcloud2, metric='euclidean')
        return min(D.flatten())

    @cached_property
    def time_filter(self):
        topo = self.all_topology
        keep_idx = {}
        for i in range(self.tp_max):
            if i == 0: keep_idx[i] = self.comparative_filter(topo[i], topo[i+1])
            elif i == self.tp_max-1: keep_idx[i] = self.comparative_filter(topo[i], topo[i-1])
            else:
                keep_prev = self.comparative_filter(topo[i], topo[i-1])
                keep_next = self.comparative_filter(topo[i], topo[i+1])
                keep_idx[i] = list(set(keep_prev) | set(keep_next))
        return keep_idx

    def comparative_filter(self, topo_i, topo_j):
        keep_idx = []
        for i, pos_i in topo_i.position.items():
            for j, pos_j in topo_j.position.items():
                if self.min_pointclould_distance(pos_i, pos_j) < self.threshold:
                    keep_idx.append(i)
        return list(set(keep_idx))

    def save_filtered(self):
        for tp in range(self.tp_max):
            self.write_pickle(tp)
        
    def write_pickle(self, tp):
        with open(f'{self.path}/{self.name}_tp{tp+1}.{self.topo_type}', 'wb') as f:
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
        return f'{self.extension}seq{self.threshold}'



if __name__ == '__main__':
    path = 'movie/test'
    import os
    for name in os.listdir(path):
        print(name)
        tp_max = len(os.listdir(f'{path}/{name}/pred'))
        SequentialFiltering(f'{path}/{name}/cmp', f"pred-0.7-semi-40_{name.replace('LI_', '')}", 'cmp', tp_max=tp_max, threshold=5).save_filtered()
        # SequentialFiltering(f'{path}/{name}/cyc', f'pred0.7_{name}', 'cyc', tp_max=tp_max, threshold=3).save_filtered()
