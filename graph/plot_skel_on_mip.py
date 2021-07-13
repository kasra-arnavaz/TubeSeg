import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
import tifffile as tif
from matplotlib import cm
from abc import ABC, abstractmethod

from graph.nx_graph import NxGraph, Cycle, Component
from utils.unpickle import read_pickle
from time_filtering.seq_filtering import SequentialFiltering
from time_filtering.trackpy_filtering import TrackpyFiltering

class PlotSkelOnMIP(ABC):

    def __init__(self, graph_path, graph_name, mip_path, mip_name):
        self.graph_path = graph_path
        self.graph_name = graph_name
        self.mip_path = mip_path
        self.mip_name = mip_name
        self.graph = NxGraph(graph_path, graph_name)

    def get_mip_figure(self):
        mip = tif.imread(f'{self.mip_path}/{self.mip_name}.tif')
        # return plt.imshow(np.flipud(mip), cmap='gray')
        return plt.imshow(np.flipud(np.rot90(mip,1)), cmap='gray')

    @abstractmethod
    def get_skel_figure(self):
        pass

    @abstractmethod
    def get_figure_title(self):
        pass

    def get_figure(self):
        plt.figure(figsize=(10,10))
        self.get_mip_figure()
        self.get_skel_figure()
        self.get_figure_title()
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()
    
    def show_figure(self):
        self.get_figure()
        plt.show()
    
    def save_figure(self, path):
        self.get_figure()
        os.makedirs(f'{path}/png', exist_ok=True)
        os.makedirs(f'{path}/pdf', exist_ok=True)
        plt.savefig(f'{path}/png/{self.graph_name}.png')
        plt.savefig(f'{path}/pdf/{self.graph_name}.pdf')
        plt.close()

class PlotComponentOnMIP(PlotSkelOnMIP):

    def __init__(self, graph_path, graph_name, mip_path, mip_name, cmp_path, cmp_name, cmp_extension):
        super().__init__(graph_path, graph_name, mip_path, mip_name)
        self.cmp = read_pickle(cmp_path, cmp_name, cmp_extension)

    def get_skel_figure(self):
        fig = nx.draw_networkx_edges(self.graph.define_graph(), pos=self.graph.nodes_xy_position(), 
                edge_color='r', width=2, edgelist=self.cmp.edge_list)
        return fig

    def get_figure_title(self):
        return plt.title(f'{self.graph_name}_#cmp{len(self.cmp.center)}')

class PlotCycleOnMIP(PlotSkelOnMIP):
    
    def __init__(self, graph_path, graph_name, mip_path, mip_name, cyc_path, cyc_name, cyc_extension):
        super().__init__(graph_path, graph_name, mip_path, mip_name)
        self.cyc = read_pickle(cyc_path, cyc_name, cyc_extension)


    def get_figure_title(self):
        return plt.title(f'{self.graph_name}_#cyc{len(self.cyc.center)}')

    def get_skel_figure(self):
        fig = nx.draw_networkx_edges(self.graph.define_graph(), pos=self.graph.nodes_xy_position(), 
                edge_color='r', width=2, edgelist=self.cyc.edge_list)
        return fig

class PlotColoredCycleOnMIP(PlotCycleOnMIP):

    def get_skel_figure(self):
        colors = cm.Paired.colors
        for edges, loop_id in zip(self.cyc.topology_edges, self.cyc.loop_id):
            nx.draw_networkx_edges(self.graph.define_graph(), pos=self.graph.nodes_xy_position(), 
                edge_color=colors[loop_id%len(colors)], width=3, edgelist=edges)
        return plt.gcf()

class PlotCmpCycOnMIP(PlotSkelOnMIP):
    
    def __init__(self, graph_path, graph_name, mip_path, mip_name, cyc_path, cyc_name, cyc_extension, cmp_path, cmp_name, cmp_extension):
        super().__init__(graph_path, graph_name, mip_path, mip_name)
        self.cyc = read_pickle(cyc_path, cyc_name, cyc_extension)
        self.cmp = read_pickle(cmp_path, cmp_name, cmp_extension)

    def get_figure_title(self):
        return plt.title(f'{self.graph_name}_#cyc{len(self.cyc.center)}_#cmp{len(self.cmp.center)}')

    def get_skel_figure(self):
        nx.draw_networkx_edges(self.graph.define_graph(), pos=self.graph.nodes_xy_position(), 
                edge_color='r', width=3, edgelist=self.cmp.edge_list)
        nx.draw_networkx_edges(self.graph.define_graph(), pos=self.graph.nodes_xy_position(), 
                edge_color='g', width=3, edgelist=self.cyc.edge_list)
        return plt.gcf()

    
if __name__ == '__main__':
    path = 'movie/val'
    for name in os.listdir(path):
        for i in range(len(os.listdir(f'{path}/{name}/pred'))):
            PlotColoredCycleOnMIP(f'{path}/{name}/pred', f"pred-0.7-semi-40_{name.replace('LI_', '')}_tp{i+1}",
                    f'{path}/{name}/mip', f"duct-mip_{name.replace('LI_', '')}_tp{i+1}",
                    f'{path}/{name}/cyc/srch=10, mem=1, thr=5, step=0.9, stop=3', f"pred-0.7-semi-40_{name.replace('LI_', '')}_tp{i+1}", 'cyctpy')\
                    .save_figure(f'{path}/{name}/cyc/srch=10, mem=1, thr=5, step=0.9, stop=3/cyc_mip')