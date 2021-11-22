import numpy as np
import pickle
import networkx as nx
from cached_property import cached_property
import os

from tracking.trackpy import Tracking
from tracking.tracking_data import Prediction
from utils.unpickle import read_pickle
from graph.nx_graph import NxGraph, Cycle, Component

class TrackpyFiltering:
    
    def __init__(self, data, track):
        self.data = data
        self.track = track
        self.sub_folder = f'srch={self.track.search_range}, mem={self.track.memory}, thr={self.track.thr}, step={self.track.step}, stop={self.track.stop}'
        os.makedirs(f'{self.data.path}/{self.sub_folder}', exist_ok=True)

    @cached_property
    def filtered_tracks(self):
        return self.track.filter_tracks()


    def write_centers_as_csv(self):
        self.filtered_tracks.to_csv(f'{self.data.path}/{self.sub_folder}/center_{self.data.name}.csv', index=False)

    @property
    def time_filter(self):
        filtered_tracks = self.filtered_tracks
        keep_idx = {}
        for t in range(self.data.tp_max):
            keep_idx_t = []
            for filter_center in filtered_tracks[filtered_tracks['frame'] == t+1].iloc[:, [0,1,2]].to_numpy():   
                keep_idx_t.append(int(np.argwhere(np.all(self.data.all_centers[t+1] == filter_center, axis=1))))
            keep_idx[t+1] = keep_idx_t
        return keep_idx

    @property
    def loop_ids(self):
        filtered_tracks = self.filtered_tracks
        loop_id = {}
        for t in range(self.data.tp_max):
            loop_id[t+1] = filtered_tracks[filtered_tracks['frame'] == t+1].iloc[:, 4].to_list()
        return loop_id

    def save_filtered(self):
        for tp in range(self.data.tp_max):
            self.write_pickle(tp)
        
    def write_pickle(self, tp):
        if f'{self.data.name}_tp{tp+1}.{self.topo_type}' not in os.listdir(f'{self.data.path}/{self.sub_folder}'):
            with open(f'{self.data.path}/{self.sub_folder}/{self.data.name}_tp{tp+1}.{self.topo_type}', 'wb') as f:
                pickle.dump(self.set_all_properties(tp), f)
    
    def set_all_properties(self, tp):
        topo = self.data.all_cycles[tp]
        keep_idx = self.time_filter[tp+1]
        sort_idx = np.argsort(keep_idx)
        self.loop_id = np.array(self.loop_ids[tp+1])[sort_idx]
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
        return f'cyctpy'


if __name__ == '__main__':
    path = 'movie/test'
    for name in os.listdir(path):
        name = 'LI_2018-11-20_emb7_pos4'
    # for name in ['LI_2019-02-05_emb5_pos4', 'LI_2018-11-20_emb7_pos4', 'LI_2018-11-20_emb6_pos1']:
        print(name)
        tp_max = len([file for file in os.listdir(f'{path}/{name}/cyc') if file.endswith('.cyc')])
        data = Prediction(f'{path}/{name}/cyc', f"pred-0.7-semi-40_{name.replace('LI_', '')}", tp_max)
        track = Tracking(data, search_range=15, memory=1, thr=10, step=0.9, stop=5)
        TrackpyFiltering(data, track).save_filtered()
        TrackpyFiltering(data, track).write_centers_as_csv()
        break