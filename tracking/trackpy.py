import matplotlib.pyplot as plt
import trackpy as tp
from tracking.tracking_data import TrackingData, Silja, Prediction
from graph.nx_graph import Cycle, NxGraph, Component

class Tracking:

    def __init__(self, data: TrackingData, search_range, memory, thr, step=None, stop=None):
        self.data = data
        self.search_range = search_range
        self.memory = memory
        self.thr = thr
        self.step = step
        self.stop = stop
        
    def track_centers(self):
        features = self.data.get_data()[['z', 'y', 'x','frame']]
        tp.quiet()
        return tp.link(features, self.search_range, memory=self.memory, adaptive_stop=self.stop, adaptive_step=self.step)

    def write_tracking_as_csv(self):
        tracked_centers = self.track_centers()
        tracked_centers.to_csv(f'{self.data.path}/track_{self.data.name}.csv')

    def filter_tracks(self):
        return tp.filtering.filter_stubs(self.track_centers(), threshold=self.thr)

    def write_filtered_tracks_as_csv(self):
        self.filter_tracks().to_csv(f'{self.data.path}/fltrtrack_{self.data.name}.csv')

    def plot_filtered_trajectory(self, ):
        plt.title(f'search_range={self.search_range}, memory={self.memory}, threshold={self.thr}, \n \
             adaptive_stop={self.stop}, adaptive_step={self.step}')
        tp.plot_traj(self.filter_tracks())

if __name__ == '__main__':
    data = Prediction('alaki2', 'one-ten_40_val_pred0.7_LI_2018-11-20_emb7_pos4', 5)
    # data.plot_data()
    track = Tracking(data, search_range=10, memory=1, thr=5, step=0.9, stop=3)
    print(track.filter_tracks())
    # print(track.track_centers())
