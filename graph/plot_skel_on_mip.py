import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
import tifffile as tif
from matplotlib import cm
from abc import ABC, abstractmethod
import pandas as pd

from graph.nx_graph import NxGraph, Cycle, Component
from utils.unpickle import read_pickle
from time_filtering.seq_filtering import SequentialFiltering
from time_filtering.trackpy_filtering import TrackpyFiltering

class PlotSkelOnMIP(ABC):

    def __init__(self, graph_path, graph_name, mip_file):
        self.mip = tif.imread(mip_file)
        self.graph_name = graph_name
        self.graph = NxGraph(graph_path, graph_name)
        self.pos = {}
        for k, (y,x) in self.graph.nodes_xy_position().items():
            self.pos[k] = [x, y]

    def get_mip_figure(self):
        return plt.imshow(self.mip, cmap='gray')
        # x = np.zeros([1024,1024])
        # return plt.imshow(x, cmap='gray')

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
        if f'{self.graph_name}.png' not in f'{path}/png':
            self.get_figure()
            os.makedirs(f'{path}/png', exist_ok=True)
            os.makedirs(f'{path}/pdf', exist_ok=True)
            plt.savefig(f'{path}/png/{self.graph_name}.png')
            plt.savefig(f'{path}/pdf/{self.graph_name}.pdf')
            plt.close()

class PlotComponentOnMIP(PlotSkelOnMIP):

    def __init__(self, graph_path, graph_name, mip_file, cmp_file):
        super().__init__(graph_path, graph_name, mip_file)
        self.cmp = read_pickle(cmp_file)

    def get_skel_figure(self):
        fig = nx.draw_networkx_edges(self.graph.define_graph(), pos=self.pos, 
                edge_color='r', width=2, edgelist=self.cmp.edge_list)
        return fig

    def get_figure_title(self):
        return plt.title(f'{self.graph_name}_#cmp{len(self.cmp.center)}')

class PlotCycleOnMIP(PlotSkelOnMIP):
    
    def __init__(self, graph_path, graph_name, mip_file, cyc_file):
        super().__init__(graph_path, graph_name, mip_file)
        self.cyc = read_pickle(cyc_file)

    def get_figure_title(self):
        return plt.title(f'{self.graph_name}_#cyc{len(self.cyc.center)}')

    def get_skel_figure(self):
        fig = nx.draw_networkx_edges(self.graph.define_graph(), pos=self.pos, 
                edge_color='r', width=2, edgelist=self.cyc.edge_list)
        return fig

class PlotColoredCycOnMIP(PlotCycleOnMIP):

    def get_skel_figure(self):
        colors = cm.Paired.colors
        for edges, loop_id in zip(self.cyc.topology_edges, self.cyc.loop_id):
            nx.draw_networkx_edges(self.graph.define_graph(), pos=self.pos, 
                edge_color=colors[loop_id%len(colors)], width=3, edgelist=edges)
        return plt.gcf()

class PlotCmpCycOnMIP(PlotSkelOnMIP):
    
    def __init__(self, graph_path, graph_name, mip_file, cyc_file, cmp_file):
        super().__init__(graph_path, graph_name, mip_file)
        self.cyc = read_pickle(cyc_file)
        self.cmp = read_pickle(cmp_file)

    def get_figure_title(self):
        return plt.title(f'{self.graph_name}_#cyc{len(self.cyc.center)}_#cmp{len(self.cmp.center)}')

    def get_skel_figure(self):
        nx.draw_networkx_edges(self.graph.define_graph(), pos=self.pos, 
                edge_color='r', width=3, edgelist=self.cmp.edge_list)
        nx.draw_networkx_edges(self.graph.define_graph(), pos=self.pos, 
                edge_color='g', width=3, edgelist=self.cyc.edge_list)
        return plt.gcf()

class CompareToSilja(PlotColoredCycOnMIP):

    def __init__(self, graph_path, graph_name, mip_file, cyc_file, silja_file, tp):
        super().__init__(graph_path, graph_name, mip_file, cyc_file)
        silja = pd.read_csv(silja_file)
        self.silja = silja.loc[silja.frame == tp, ['x', 'y', 'loop_id']]

    def silja_process(self):
        S, pos, loop_id = nx.Graph(), {}, []
        for index, row in self.silja.iterrows():
            S.add_node(index)
            *coor, id = row.to_list()
            pos[index] = coor
            loop_id.append(id)
        return S, pos, np.array(loop_id)

    def get_skel_figure(self):
        colors = np.delete(np.array(list(cm.tab10.colors)), -3, 0)
        silja_graph, silja_pos, silja_id = self.silja_process()
        for edges, loop_id in zip(self.cyc.topology_edges, self.cyc.loop_id):
            nx.draw_networkx_edges(self.graph.define_graph(), pos=self.pos, 
                edge_color=colors[loop_id%len(colors)], width=3, edgelist=edges)
        silja_colors = colors[np.remainder(silja_id,len(colors)).astype(int)]
        nx.draw_networkx_nodes(silja_graph, pos=silja_pos, node_color=silja_colors, node_shape='x', linewidths=3)
        return plt.gcf()

    def get_figure_title(self):
        return plt.title(f'{self.graph_name}_#truecyc{len(self.silja_process()[-1])}_#predcyc{len(self.cyc.center)}')




    
if __name__ == '__main__':
    path = 'movie/test'
    for name in os.listdir(path):
        print(name)
    # for name in ['LI_2019-02-05_emb5_pos4', 'LI_2018-11-20_emb7_pos4', 'LI_2018-11-20_emb6_pos1']:
    # for name in ['LI_2019-07-03_emb7_pos2', 'LI_2019-07-03_emb7_pos3', 'LI_2019-07-03_emb7_pos4']:
    #     name = 'LI_2019-07-03_emb7_pos2'
        tp_max = len(os.listdir(f'{path}/{name}/pred'))
        for t in range(1, tp_max+1):
            PlotComponentOnMIP(f'{path}/{name}/pred', f"pred-0.7-semi-40_{name.replace('LI_', '')}_tp{t}",
                            f"{path}/{name}/mip/duct-mip_{name.replace('LI_', '')}_tp{t}.tif",
                            f"{path}/{name}/cmp/pred-0.7-semi-40_{name.replace('LI_', '')}_tp{t}.cmp")\
                        .save_figure(f'{path}/{name}/cmp')
            # PlotColoredCycOnMIP(f'{path}/{name}/pred', f"pred-0.7-semi-40_{name.replace('LI_', '')}_tp{t}",
            #         f"{path}/{name}/mip/duct-mip_{name.replace('LI_', '')}_tp{t}.tif",
            #         f"{path}/{name}/cyc/srch=15, mem=1, thr=15, step=0.9, stop=5/pred-0.7-semi-40_{name.replace('LI_', '')}_tp{t}.cyctpy")\
            #             .save_figure(f'{path}/{name}/cyc/srch=15, mem=1, thr=15, step=0.9, stop=5')
            # CompareToSilja(f'{path}/{name}/pred', f"pred-0.7-semi-40_{name.replace('LI_', '')}_tp{t}",
            #         f"{path}/{name}/mip/duct-mip_{name.replace('LI_', '')}_tp{t}.tif",
            #         f"{path}/{name}/cyc/srch=15, mem=1, thr=15, step=0.9, stop=5/pred-0.7-semi-40_{name.replace('LI_', '')}_tp{t}.cyctpy",
            #         f'tracking/silja/{name}_test.csv', t)\
            #             .save_figure(f'{path}/{name}/cyc/srch=15, mem=1, thr=15, step=0.9, stop=5')