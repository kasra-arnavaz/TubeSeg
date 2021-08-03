from abc import ABC, abstractproperty
import numpy as np
import pandas as pd

from graph.nx_graph import Component, Cycle, NxGraph
from utils.unpickle import read_pickle
from time_filtering.trackpy_filtering import TrackpyFiltering

class TopologyScore(ABC):

    def __init__(self, label_path, label_name, pred_path, pred_name, matching_radius=10, zero_thr=0.3, one_thr=0.7):
        self.label_path = label_path
        self.label_name = label_name
        self.pred_path = pred_path
        self.pred_name =  pred_name
        self.matching_radius = matching_radius
        self.zero_thr = zero_thr
        self.one_thr = one_thr
    
    @abstractproperty
    def label_topology(self):
        pass

    @abstractproperty
    def prediction_topology(self):
        pass

    @abstractproperty
    def topology_type(self):
        pass

    @staticmethod
    def euclidean(A, B):
        '''A is (a x n) and B is (b x n);
        returns a matrix (a x b) whose elements
        are the euclidean distances between points.'''
        p1 = np.sum(A**2, 1)[:, np.newaxis]
        p2 = np.sum(B**2, 1)
        p3 = -2*np.dot(A,B.T)
        return np.sqrt(p1+p2+p3)

    @property
    def patch_id(self):
        return self.label_name[-2:]

    def get_allowed_range(self):
        patch_map = np.array(['A1','B1','C1','D1',
                    'A2','B2','C2','D2',
                    'A3','B3','C3','D3',
                    'A4','B4','C4','D4'])
        index = int(np.argwhere(patch_map == self.patch_id))
        col, row = np.unravel_index(index, (4,4))
        return [col*256, (col+1)*256-1], [row*256, (row+1)*256-1]

    def is_in_patch(self, pos):
        y_range, x_range = self.get_allowed_range()
        y, x = pos[:,1], pos[:,2]
        return all((y <= y_range[1]) & (y>=y_range[0]) & (x <= x_range[1]) & (x>=x_range[0])) 

    def match_nodes(self):
        tp_label, tp_pred, fp, fn = np.zeros([4, len(self.label_topology.position), len(self.pred_pos)])
        for i, label_pos in self.label_topology.position.items():
            for j, pred_pos in enumerate(self.pred_pos.values()):
                distance_matrix = self.euclidean(label_pos, pred_pos)
                label_nearest_neighbor_distance = np.amin(distance_matrix, 1)
                pred_nearest_neighbor_distance = np.amin(distance_matrix, 0)
                tp_label[i,j] = np.sum(label_nearest_neighbor_distance <= self.matching_radius)
                tp_pred[i,j] = np.sum(pred_nearest_neighbor_distance <= self.matching_radius)
                fp[i,j] = distance_matrix.shape[1] - tp_pred[i,j]
                fn[i,j] = distance_matrix.shape[0] - tp_label[i,j]
        return tp_label, tp_pred, fp, fn
    
    def normalized_iou(self):
        tp_label, tp_pred, fp, fn = self.match_nodes()
        n1 = tp_label / (tp_label+fn)
        n2 = tp_pred / (tp_pred+fp)
        d1 = (tp_label+2*fn) / (tp_label+fn)
        d2 = (tp_pred+2*fp) / (tp_pred+fp)
        return (n1+n2)/(d1+d2)

    def write_normalized_iou(self):
        normalized_iou = self.normalized_iou()
        data_frame = pd.DataFrame(np.round(normalized_iou,2),
                        index=range(normalized_iou.shape[0]), columns=range(normalized_iou.shape[1]))
        data_frame.to_csv(f'{self.pred_path}/{self.topology_type}_{self.pred_name}.csv')

    def filtered_normalized_iou(self):
        normalized_iou = self.normalized_iou()
        zero_mask = normalized_iou <= self.zero_thr
        one_mask = normalized_iou >= self.one_thr
        normalized_iou[zero_mask] = 0
        normalized_iou[one_mask] = 1
        return normalized_iou
    
    def row_and_column_scores(self):
        filtered_normalized_iou = self.filtered_normalized_iou()
        # making sure the scores don't go over 1.
        row_score = np.minimum(1, np.sum(filtered_normalized_iou, 1))
        column_score = np.minimum(1, np.sum(filtered_normalized_iou, 0))
        return row_score, column_score

    def mean_row_and_column_scores(self):
        self.pred_pos = {j: pred_pos for j, pred_pos in self.prediction_topology.position.items() if self.is_in_patch(pred_pos)}
        label_len, pred_len = len(self.label_topology.position), len(self.pred_pos)
        print(label_len, pred_len)
        if label_len > 0 and pred_len > 0:
            row_score, column_score = self.row_and_column_scores()
            mean_row_score, mean_column_score = row_score.mean(), column_score.mean()
        elif label_len == 0 and pred_len > 0:
            mean_row_score, mean_column_score = 1/(1+pred_len), 1/(1+pred_len)
        elif label_len > 0 and pred_len == 0:
            mean_row_score, mean_column_score = 1/(1+label_len), 1/(1+label_len)
        elif label_len ==0 and pred_len==0:
            mean_row_score, mean_column_score = 1, 1
        return mean_row_score, mean_column_score
    
    def final_score(self):
        mean_row_score, mean_column_score = self.mean_row_and_column_scores()
        return (mean_row_score + mean_column_score)/2


class ComponentScore(TopologyScore):

    @property
    def label_topology(self):
        return Component(self.label_path, self.label_name)
    
    @property
    def prediction_topology(self):
        return Component(self.pred_path, self.pred_name)

    @property
    def topology_type(self):
        return 'cmp'

class CycleScore(TopologyScore):

    @property
    def label_topology(self):
        return Cycle(self.label_path, self.label_name)

    @property
    def prediction_topology(self):
        return Cycle(self.pred_path, self.pred_name)

    @property
    def topology_type(self):
        return 'cyc'


class PickledCycleScore(TopologyScore):

    @property
    def label_topology(self):
        return read_pickle(f'{self.label_path}/{self.label_name}.cyc')

    @property
    def prediction_topology(self):
        return read_pickle(f'{self.pred_path}/{self.pred_name}.cyctpy')

    @property
    def topology_type(self):
        return 'cyctpy'





if __name__ == '__main__':

    # cmp_score = ComponentScore('alaki', 'seg_LI_2019-09-19_emb1_pos1_tp264_A4_B4',
    #                             'alaki', 'one-ten_40_val_pred0.7_LI_2019-09-19_emb1_pos1_tp264_A4_B4')
    # cyc_score = CycleScore('alaki', 'seg_LI_2019-09-19_emb1_pos1_tp264_A4_B4',
    #                             'alaki', 'one-ten_40_val_pred0.7_LI_2019-09-19_emb1_pos1_tp264_A4_B4')
    # cmp_score.write_normalized_iou()
    # cyc_score.write_normalized_iou()
    # print(cmp_score.final_score())
    # print(cyc_score.final_score())

    cyc_score = PickledCycleScore('D:/dataset/test/patches/cyc', 'label_ts_LI-2018-11-20-emb6-pos1_tp104-D1_A1',
                                 'movie/test/LI_2018-11-20_emb6_pos1/cyc/srch=15, mem=1, thr=20, step=0.9, stop=5',
                                 'pred-0.7-semi-40_2018-11-20_emb6_pos1_tp104')

    print(cyc_score.final_score())