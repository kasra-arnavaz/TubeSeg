import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from cached_property import cached_property

class HOTA:

    def __init__(self, gt_path, gt_name, pr_path, pr_name, thr):
        self.gt_data = pd.read_csv(f'{gt_path}/{gt_name}.csv')
        self.pr_data = pd.read_csv(f'{pr_path}/{pr_name}.csv')
        self.num_frame = int(np.maximum(np.amax(self.gt_data['frame']), np.amax(self.pr_data['frame'])))
        self.thr = thr

    def get_frame_data(self, frame):
        mask_gt = self.gt_data['frame'] == frame
        mask_pr = self.pr_data['frame'] == frame
        return self.gt_data[mask_gt], self.pr_data[mask_pr]

    def get_frame_idx(self, frame):
        return self.get_frame_data(frame)[0].index.to_numpy(), self.get_frame_data(frame)[1].index.to_numpy()

    def get_frame_pos(self, frame):
        gt_frame, pr_frame = self.get_frame_data(frame)
        return gt_frame.loc[:, ['x', 'y']].to_numpy(), pr_frame.loc[:, ['x', 'y']].to_numpy()

    def get_frame_ID(self, frame):
        gt_frame, pr_frame = self.get_frame_data(frame)
        return gt_frame.iloc[:, -1], pr_frame.iloc[: ,-1]

    def get_num_detections(self, frame):
        return self.get_frame_data(frame)[0].shape[0], self.get_frame_data(frame)[1].shape[0]

    @staticmethod
    def euclidean(A, B):
        p1 = np.sum(A**2, 1)[:, np.newaxis]
        p2 = np.sum(B**2, 1)
        p3 = -2*np.dot(A,B.T)
        return np.sqrt(p1+p2+p3)

    def similarity_measure(self, frame):
        distance = self.euclidean(*self.get_frame_pos(frame))
        gt_idx_fr, pr_idx_fr = linear_sum_assignment(distance)
        gt_idx, pr_idx = self.get_frame_idx(frame)[0][gt_idx_fr], self.get_frame_idx(frame)[1][pr_idx_fr]
        return distance[gt_idx_fr, pr_idx_fr], gt_idx, pr_idx

    def detection_frame(self, frame):
        S = self.similarity_measure(frame)[0]
        tp = (S<=self.thr).sum()
        fn = self.get_num_detections(frame)[0] - tp
        fp = self.get_num_detections(frame)[1] - tp
        return tp, fn, fp

    @cached_property
    def DetA(self): # detection accuracy (IoU, Jacaard)
        DetA_frame = []
        for frame in range(1, self.num_frame+1):
            tp, fn, fp = self.detection_frame(frame)
            DetA_frame.append(tp/(tp+fn+fp))
        return sum(DetA_frame)/len(DetA_frame)

    def match_idx(self, frame):
        S, gt_idx, pr_idx = self.similarity_measure(frame)
        match = (S<=self.thr)
        gt_match, pr_match = gt_idx[match], pr_idx[match]
        return gt_match, pr_match

    @cached_property
    def tp_idx_all(self):
        gt_all_idx, pr_all_idx = [], []
        for frame in range(1, self.num_frame+1):
            gt_idx, pr_idx = self.match_idx(frame)
            gt_all_idx.append(gt_idx)
            pr_all_idx.append(pr_idx)
        gt_all_idx = [ll for l in gt_all_idx for ll in l]
        pr_all_idx = [ll for l in pr_all_idx for ll in l]
        return gt_all_idx, pr_all_idx

    @cached_property
    def fn_idx_all(self):
        all_idx_gt = self.gt_data.index.to_list()
        all_idx_tp_gt = self.tp_idx_all[0]
        all_idx_fn = list(set(all_idx_gt) - set(all_idx_tp_gt))
        return all_idx_fn

    @cached_property
    def fp_idx_all(self):
        all_idx_pr = self.pr_data.index.to_list()
        all_idx_tp_pr = self.tp_idx_all[1]
        all_idx_fp = list(set(all_idx_pr) - set(all_idx_tp_pr))
        return all_idx_fp

    @cached_property
    def AssA(self):
        tpa, fna, fpa = 0, 0, 0
        A_c = []
        for gt_idx_c, pr_idx_c in zip(*self.tp_idx_all):
            gt_ID_c, pr_ID_c = self.gt_data.loc[gt_idx_c, 'loop_id'], self.pr_data.loc[pr_idx_c, 'particle']
            for gt_idx_k, pr_idx_k in zip(*self.tp_idx_all):
                gt_ID_k, pr_ID_k = self.gt_data.loc[gt_idx_k, 'loop_id'], self.pr_data.loc[pr_idx_k, 'particle']
                if (gt_ID_c == gt_ID_k) and (pr_ID_c == pr_ID_k): tpa += 1
                if (gt_ID_c == gt_ID_k) and (pr_ID_c != pr_ID_k): fna += 1
                if (pr_ID_c == pr_ID_k) and (gt_ID_c != gt_ID_k): fpa += 1
            for gt_idx_k in self.fn_idx_all:
                gt_ID_k = self.gt_data.loc[gt_idx_k, 'loop_id']
                if (gt_ID_c == gt_ID_k): fna += 1
            for pr_idx_k in self.fp_idx_all:
                pr_ID_k = self.pr_data.loc[pr_idx_k, 'particle']
                if (pr_ID_c == pr_ID_k): fpa += 1
            A_c.append(tpa/(tpa+fna+fpa))
        return sum(A_c)/len(A_c)

    @property
    def HOTA(self):
        return np.sqrt(self.AssA*self.DetA)
    
    def print(self):
        print(f'DetA = {self.DetA}, AssA = {self.AssA}, HOTA = {self.HOTA}')



if __name__ == '__main__':
    HOTA('.', 'LI 2018-12-07_emb6_pos4_val', 'movie/val/LI_2018-12-07_emb6_pos4/cyc/srch=10, mem=0, thr=20, step=0.9, stop=3',\
         'center_pred-0.7-semi-40_2018-12-07_emb6_pos4', 10).print()
    # HOTA('tracking/hota_sanity', 'gt_sanity', 'tracking/hota_sanity', 'pr_sanity_c', 1).print()

