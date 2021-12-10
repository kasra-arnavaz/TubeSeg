from abc import ABC, abstractmethod, abstractproperty
from cached_property import cached_property
from argparse import ArgumentParser
import itertools
import pandas as pd
import networkx as nx
import itertools
import pickle
import os

from topology.unpickle import read_pickle

class NxGraph:
    ''' Loads the .skel file as a graph in NetworkX module.
    '''
    def __init__(self, skel_path: str, name: str):
        self.skel_path = skel_path
        self.name = name

    def load_skeleton(self):
        return pd.read_table(f'{self.skel_path}/{self.name}.skel', sep=' ', header=None, usecols=[0,1,2,3])

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

    def nodes_yx_position(self):
        pos = {}
        for node, coor in self.get_nodes_and_edges()[0].iterrows():
            pos[node] = coor[1:3].tolist()
        return pos


class Topology(ABC):

    def __init__(self, skel_path, name, write_path: str = None):
        if write_path is None: write_path = f'{skel_path}/..'
        self.skel_path = skel_path
        self.name = name
        self.write_path = f'{write_path}/{self.topo_type}'
        self.graph = NxGraph(skel_path, name)

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
    def _nodes_yx_position(self):
        return self.graph.nodes_yx_position()

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
    def _center_yx(self):
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
        self.nodes_yx_position = self._nodes_yx_position
        self.center = self._center
        self.center_yx = self._center_yx
        self.len = self.__len__()
        return self

    def write_pickle(self):
        os.makedirs(f'{self.write_path}', exist_ok=True)
        if f'{self.name}.{self.topo_type}' not in os.listdir(self.write_path):
            with open(f'{self.write_path}/{self.name}.{self.topo_type}', 'wb') as f:
                pickle.dump(self.set_all_properties(), f)
        elif os.path.getsize(f'{self.write_path}/{self.name}.{self.topo_type}') < 1000: # if file is aborted
            with open(f'{self.write_path}/{self.name}.{self.topo_type}', 'wb') as f:
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
    parser.add_argument('--skel_path', type=str)
    args = parser.parse_args()
    names = [name.replace('.skel', '') for name in os.listdir(args.skel_path) if name.endswith('.skel')]
    for name in names:
        print(name)
        Component(args.skel_path, name).write_pickle()
        Cycle(args.skel_path, name).write_pickle()