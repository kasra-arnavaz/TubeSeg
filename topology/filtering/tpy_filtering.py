import numpy as np
import pickle
import networkx as nx
from cached_property import cached_property
from argparse import ArgumentParser
import os

from topology.tracking.trackpy import Tracking
from topology.tracking.tracking_data import TrackingData, Prediction
from topology.unpickle import read_pickle
from topology.cyc_cmp import NxGraph, Cycle, Component


class TrackpyFiltering:
    '''Filters the cycles which last a short while and saved the filtered cycles as pickle file in write_path.
    '''
    def __init__(self, data: TrackingData, track: Tracking, write_path: str = None):
        if write_path is None: write_path = data.cyc_path
        self.data = data
        self.track = track
        self.write_path = write_path

    @cached_property
    def filtered_tracks(self):
        return self.track.filter_tracks()

    def write_centers_as_csv(self):
        self.filtered_tracks.to_csv(f'{self.data.path}/center_{self.data.name}.csv', index=False)

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
        if f'{self.data.name}_tp{tp+1}.{self.topo_type}' not in os.listdir(self.write_path):
            with open(f'{self.write_path}/{self.data.name}_tp{tp+1}.{self.topo_type}', 'wb') as f:
                pickle.dump(self.set_all_properties(tp), f)
        elif os.path.getsize(f'{self.write_path}/{self.data.name}_tp{tp+1}.{self.topo_type}') < 1000: 
            with open(f'{self.write_path}/{self.data.name}_tp{tp+1}.{self.topo_type}', 'wb') as f:
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
        self.center_yx = {key:value for key, value in topo.center_yx.items() if key in keep_idx}
        self.node_list = nx.utils.misc.flatten(self.topology_nodes)
        self.edge_list = [edge for edge in topo.edge_list if (edge[0] in self.node_list) and (edge[1] in self.node_list)]
        self.nodes_yx_position = topo.nodes_yx_position
        return self

    @property
    def topo_type(self):
        return f'cyctpy{self.track.thr}'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cyc_path', type=str)
    parser.add_argument('--cyc_name', type=str)
    parser.add_argument('--search_range', type=int, default=15)
    parser.add_argument('--memory', type=int, default=1)
    parser.add_argument('--thr', type=int, default=15)
    parser.add_argument('--step', type=float, default=0.9)
    parser.add_argument('--stop', type=int, default=5)
    args = parser.parse_args()
    data = Prediction(args.cyc_path, args.cyc_name)
    track = Tracking(data, args.search_range, args.memory, args.thr, args.step, args.stop)
    TrackpyFiltering(data, track).save_filtered()
