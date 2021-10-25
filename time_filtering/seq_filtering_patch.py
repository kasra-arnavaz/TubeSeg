from abc import ABC, abstractmethod
from numpy.core.fromnumeric import cumprod
from scipy.spatial.distance import cdist
import pickle
import networkx as nx
from cached_property import cached_property
import os

from graph.nx_graph import Topology, Component, Cycle, NxGraph
from utils.unpickle import read_pickle


class SequentialFilteringPatches:

    def __init__(self, path, threshold, current_frame_path):
        self.path = path
        self.threshold = threshold
        self.current_frame_path = current_frame_path

    @cached_property
    def all_topology(self):
        all_topo = []
        names = [name for name in os.listdir(self.current_frame_path) if name.endswith('cmp')]
        for name in names:
            name = name.replace('LI-', '')
            current_tp = int(name.split('_')[-2][2:])
            prev_name = name.replace(f'tp{current_tp}', f'tp{current_tp-1}')
            next_name = name.replace(f'tp{current_tp}', f'tp{current_tp+1}')
            print(f'{self.path}/{next_name}')
            current_topo = read_pickle(f'{self.path}/{name}')
            prev_topo = read_pickle(f'{self.path}/{prev_name}')
            next_topo = read_pickle(f'{self.path}/{next_name}')
            all_topo.append([prev_topo, current_topo, next_topo])
        return all_topo

    @staticmethod
    def min_pointclould_distance(pointcloud1, pointcloud2):
        # Pointclouds in N x 3 format
        D = cdist(pointcloud1, pointcloud2, metric='euclidean')
        return min(D.flatten())

    @cached_property
    def time_filter(self):
        topo = self.all_topology
        keep_idx = {}
        for tri_topo in topo:
            keep_prev = self.comparative_filter(tri_topo[1], tri_topo[0])
            keep_next = self.comparative_filter(tri_topo[1], tri_topo[2])
            keep_idx[tri_topo[1].name] = list(set(keep_prev) | set(keep_next))
        return keep_idx

    def comparative_filter(self, topo_i, topo_j):
        keep_idx = []
        for i, pos_i in topo_i.position.items():
            for j, pos_j in topo_j.position.items():
                if self.min_pointclould_distance(pos_i, pos_j) < self.threshold:
                    keep_idx.append(i)
        return list(set(keep_idx))
        
    def write_pickle(self):
        for i, name in enumerate(self.time_filter.keys()):
            with open(f'{self.current_frame_path}/{name}.{self.topo_type}', 'wb') as f:
                pickle.dump(self.set_all_properties(name, i), f)
    
    def set_all_properties(self, name, idx):
        keep_idx = self.time_filter[name]
        topo = self.all_topology[idx][1]
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
        return f'cmpseq{self.threshold}'



if __name__ == '__main__':
    SequentialFilteringPatches('D:/dataset/prev_next_patches_semi', 5, 'D:/dataset/prev_next_patches_semi/current_frame').write_pickle()
