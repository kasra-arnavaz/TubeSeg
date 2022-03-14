from cached_property import cached_property
from topology.filtering.tpy_filtering import TpyFiltering
from topology.tracking.hota import HOTA
from topology.tracking.trackpy import Tracking
from topology.tracking.tracking_data import Prediction
from topology.unpickle import read_pickle
from topology.cyc_cmp import Cycle, Component

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
            tp_max = len([file for file in os.listdir(f'{self.path}/{name}/cyc') if file.endswith('.cyc')])
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
                            DetA, AssA, Hota, DetRe, DetPr, AssRe, AssPr = [], [], [], [], [], [], []
                            for name, data in self.get_data.items():
                                
                                track = Tracking(data, search_range=search, memory=mem, thr=thr, step=step, stop=stop)
                                TpyFiltering(data, track).write_centers_as_csv()
                                hota = HOTA(f'{self.path}/{name}', name, f'{self.path}/{name}/cyc/srch={search}, mem={mem}, thr={thr}, step={step}, stop={stop}',\
                                f'center_{data.name}', 50)
                                hota.print()
                                DetA.append(hota.DetA[0])
                                DetRe.append(hota.DetA[1])
                                DetPr.append(hota.DetA[2])
                                AssA.append(hota.AssA[0])
                                AssRe.append(hota.AssA[1])
                                AssPr.append(hota.AssA[2])
                                Hota.append(hota.HOTA)
                            mean_DetA, mean_AssA, mean_Hota = sum(DetA)/len(DetA), sum(AssA)/len(AssA), sum(Hota)/len(Hota)
                            mean_DetRe, mean_DetPr = sum(DetRe)/len(DetRe), sum(DetPr)/len(DetPr)
                            mean_AssRe, mean_AssPr = sum(AssRe)/len(AssRe), sum(AssPr)/len(AssPr)
                            results[f'{search}, {mem}, {thr}, {step}, {stop}'] = [mean_DetA, mean_AssA, mean_Hota, mean_DetRe, mean_DetPr, mean_AssRe, mean_AssPr]
        return results

    def best_param(self, metric='hota'):
        best_measure = 0
        results = self.grid_search()
        with open(f'{self.path}/results.txt', 'w') as f:
            for k, v in results.items():
                f.write(f'{k}: \t {v}\n')
        for params, measures in results.items():
            if metric == 'hota': measure = measures[-1]
            elif metric == 'assa': measure = measures[1]
            elif metric == 'deta': measure = measures[0]
            if measure >= best_measure:
                best_measure = measure
                best_params = params

        return best_params, best_measure

if __name__ == '__main__':
    print(TuneTrackingParameters('cldice_unet/dev_silja', search_list=[5,10,15], mem_list=[1], thr_list=[10,12,15], step_list=[0.9], stop_list=[3,5]).grid_search())