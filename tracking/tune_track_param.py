from cached_property import cached_property
from time_filtering.trackpy_filtering import TrackpyFiltering
from tracking.hota import HOTA
from tracking.trackpy import Tracking
from tracking.tracking_data import Prediction
from utils.unpickle import read_pickle
from graph.nx_graph import NxGraph, Cycle, Component

import os
from cached_property import cached_property

class TuneTrackingParameters:

    def __init__(self, path, search_list, mem_list, thr_list, step_list, stop_list):
        self.search_list = search_list
        self.mem_list = mem_list
        self.thr_list = thr_list
        self.step_list = step_list
        self.stop_list = stop_list
        self.path = path

    @cached_property
    def get_data(self):
        all_data = {}
        for name in os.listdir(self.path):
            tp_max = len(os.listdir(f'{self.path}/{name}/pred'))
            data = Prediction(f'{self.path}/{name}/cyc', f"pred-0.7-semi-40_{name.replace('LI_', '')}", tp_max)
            all_data[name] = data
        return all_data

    def grid_search(self):
        results = {}
        for search in self.search_list:
            for mem in self.mem_list:
                for thr in self.thr_list:
                    for step in self.step_list:
                        for stop in self.stop_list:
                            DetA, AssA, Hota = [], [], []
                            for name, data in self.get_data.items():
                                track = Tracking(data, search_range=search, memory=mem, thr=thr, step=step, stop=stop)
                                TrackpyFiltering(data, track).write_centers_as_csv()
                                hota = HOTA(f'movie/silja/{name}', name, f'movie/silja/{name}/cyc/srch={search}, mem={mem}, thr={thr}, step={step}, stop={stop}',\
                                f'center_{data.name}', 50)
                                DetA.append(hota.DetA)
                                AssA.append(hota.AssA)
                                Hota.append(hota.HOTA)
                            mean_DetA, mean_AssA, mean_Hota = sum(DetA)/len(DetA), sum(AssA)/len(AssA), sum(Hota)/len(Hota)
                            results[f'{search}, {mem}, {thr}, {step}, {stop}'] = [mean_DetA, mean_AssA, mean_Hota]
        return results

    def best_param(self, metric='hota'):
        best_measure = 0
        results = self.grid_search()
        print(results)
        for params, measures in results.items():
            if metric == 'hota': measure = measures[-1]
            elif metric == 'assa': measure = measures[1]
            elif metric == 'deta': measure = measures[0]
            if measure >= best_measure:
                best_measure = measure
                best_params = params

        return best_params, best_measure

if __name__ == '__main__':
    print(TuneTrackingParameters('movie/silja', search_list=[50, 25 ,15, 10], mem_list=[0,1,3,5,8], thr_list=[5,10,15,20], step_list=[0.9], stop_list=[3, 5, 7, 10]).best_param())