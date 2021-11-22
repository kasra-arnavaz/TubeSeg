import itertools
import pandas as pd
import networkx as nx
import itertools
from abc import ABC, abstractmethod, abstractproperty
#from graph.skeleton import Skeleton
import pickle
from cached_property import cached_property
import time
import os
from utils.unpickle import read_pickle
from argparse import ArgumentParser

class NxGraph:

    def __init__(self, path, name):
        self.path = path
        self.name = name
        # Skeleton(path, name)

    def load_skeleton(self):
        return pd.read_table(f'{self.path}/skel_{self.name}.graph', sep=' ', header=None, usecols=[0,1,2,3])

    def get_nodes_and_edges(self):
        skel = self.load_skeleton()
        is_nan = skel.loc[:,1].astype('str').str.contains('nan')
        nodes = skel.loc[(~is_nan) & (skel[0]=='n'), 1:3].astype('float')
        edges = skel.loc[skel[0] == 'c', [1,2]].astype('int')
        return nodes, edges

    def define_graph(self):
        edges = self.get_nodes_and_edges()[1]
        edge_list = list(edges.itertuples(index=False, name=None))
        graph = nx.Graph()
        graph.add_edges_from(edge_list)
        return graph

    def nodes_xy_position(self):
        pos = {}
        for node, coor in self.get_nodes_and_edges()[0].iterrows():
            pos[node] = coor[1:3].tolist()
        return pos


class Topology(ABC):

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.graph = NxGraph(path, name)

    @abstractproperty
    def topo_type(self):
        pass

    @cached_property
    @abstractmethod    
    def _topology_nodes(self):
        pass

    @cached_property
    def _topology_edges(self):
        topo_edges = []
        for nodes in self._topology_nodes:
            topo_edges_i = []
            for edge in self.graph.define_graph().edges:
                if edge[0] in nodes and edge[1] in nodes:
                    topo_edges_i.append(edge)
            topo_edges.append(topo_edges_i)
        return topo_edges
            
    @cached_property
    def _nodes_xy_position(self):
        return self.graph.nodes_xy_position()

    @cached_property
    def _node_list(self):
        return nx.utils.misc.flatten(self._topology_nodes)

    @cached_property
    def _edge_list(self):
        return [edge for edge in self.graph.define_graph().edges if (edge[0] in self._node_list) and (edge[1] in self._node_list)]
    
    def __len__(self):
        return len(self._topology_nodes)

    @cached_property
    def _position(self):
        pos = {}
        for i, topo in enumerate(self._topology_nodes):
            pos[i] = self.graph.load_skeleton().iloc[list(topo), [1,2,3]].to_numpy(dtype='float')
        return pos
    
    @cached_property
    def _center(self):
        center = {}
        for i, pos in self._position.items():
            center[i] = pos.mean(axis=0)
        return center
    
    @cached_property
    def _center_xy(self):
        center = {}
        for i, pos in self._position.items():
            center[i] = pos.mean(axis=0)[1:3]
        return center

    def set_all_properties(self):
        self.topology_nodes = self._topology_nodes
        self.topology_edges = self._topology_edges
        self.node_list = self._node_list
        self.edge_list = self._edge_list
        self.position = self._position
        self.nodes_xy_position = self._nodes_xy_position
        self.center = self._center
        self.center_xy = self._center_xy
        self.len = self.__len__()
        return self

    def write_pickle(self, write_path=None):
        if write_path is None: write_path = self.path
        else: os.makedirs(write_path, exist_ok=True)
        if f'{self.name}.{self.topo_type}' not in os.listdir(write_path):
            with open(f'{write_path}/{self.name}.{self.topo_type}', 'wb') as f:
                pickle.dump(self.set_all_properties(), f)
        elif os.path.getsize(f'{write_path}/{self.name}.{self.topo_type}') < 100:
            with open(f'{write_path}/{self.name}.{self.topo_type}', 'wb') as f:
                pickle.dump(self.set_all_properties(), f)



class Component(Topology):

    @property
    def topo_type(self):
        return 'cmp'

    def component_generator(self):
        return nx.algorithms.components.connected_components(self.graph.define_graph())

    def nodes_in_small_components(self):
        nodes_of_large_cmp = list(itertools.chain(*self._topology_nodes))
        return list( set(self.graph.define_graph().nodes) - set(nodes_of_large_cmp) )

    def refined_graph(self):
        graph = self.graph.define_graph()
        remove_nodes = self.nodes_in_small_components()
        graph.remove_nodes_from(remove_nodes)
        return graph

    @cached_property
    def _topology_nodes(self, small_size=5):
        return [list(nodes) for nodes in self.component_generator() if len(nodes) >= small_size]


class Cycle(Topology):

    @property
    def topo_type(self):
        return 'cyc'
    
    @cached_property
    def _topology_nodes(self, small_size=5):
        topology_nodes = nx.algorithms.cycles.minimum_cycle_basis(self.graph.define_graph())
        return [list(nodes) for nodes in topology_nodes if len(nodes) >= small_size]



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--skelpath', type=str)
    args = parser.parse_args()
    #path = 'results/semi/2d/images/pred/ts/0.7/patches'
    names = [name.replace('.tif', '') for name in os.listdir(args.skelpath) if name.endswith('.tif')]
    for name in names:
        print(name)
        Component(args.skelpath, name).write_pickle(f'{args.skelpath}/../cmp')
        Cycle(args.skelpath, name).write_pickle(f'{args.skelpath}/../cyc')
