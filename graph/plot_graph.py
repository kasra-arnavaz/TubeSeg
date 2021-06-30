from graph.nx_graph import Graph, Component, Cycle
import os
import networkx as nx
import matplotlib.pyplot as plt


class PlotGraph:

    def __init__(self, path, name, show_center_cmp=True, show_center_cyc=True):
        self.path = path
        self.name = name
        self.show_center_cmp = show_center_cmp
        self.show_center_cyc = show_center_cyc
        self.graph = Graph(path, name)
        self.cyc = Cycle(path, name)
        self.cmp = Component(path, name)

    def component_center_graph(self):
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cmp)))
        return g

    def cycle_center_graph(self):
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cyc)))
        return g
    
    def null_graph(self):
        ''' making an invisible graph to prevent the actual graph
        from being stretched over the figure plane.'''
        if self.name.endswith('_'): size = 1024     #image
        else: size = 256    #patch
        null = nx.Graph()
        null.add_nodes_from([0,1,2,3])
        pos_null={0:[0,0], 1:[0,size], 2:[size,0], 3:[size,size]}
        return null, pos_null

    def get_figure(self):
        plt.title(f'{self.name}_#L{len(self.cyc)}_#C{len(self.cmp)}', y=-0.1)
        nx.draw(self.null_graph()[0], self.null_graph()[1], node_size=1e-10)
        nx.draw(self.cmp.refined_graph(), pos=self.graph.nodes_xy_position(), 
                node_size=50, node_color='r', with_labels=False)
        nx.draw(self.cmp.refined_graph(), pos=self.graph.nodes_xy_position(), 
                node_size=50, node_color='g', with_labels=False, nodelist=self.cyc.node_list())
        if self.show_center_cmp:
            nx.draw(self.component_center_graph(), pos=self.cmp.center_xy, 
                    node_size=100, node_color='k', node_shape='s', with_labels=True, font_color='w')
        if self.show_center_cyc:
            nx.draw(self.cycle_center_graph(), pos=self.cyc.center_xy,
                     node_size=1e-10, node_color='w', node_shape='s', with_labels=True, font_color='k')
        return plt.gcf()
    
    def show_figure(self):
        self.get_figure()
        plt.show()
    
    def save_figure(self):
        plt.savefig(f'{self.path}/{self.name}.png', transparent=True)
        plt.savefig(f'{self.path}/{self.name}.pdf')
        plt.close()



if __name__ == '__main__':
    PlotGraph('alaki', 'one-ten_40_val_pred0.7_LI_2019-08-30_emb2_pos1_tp162_D2D3D4_A2').show_figure()
