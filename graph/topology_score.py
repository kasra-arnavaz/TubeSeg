from abc import ABC, abstractproperty
import numpy as np
import pandas as pd

from graph.nx_graph import Component, Cycle



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

    def match_nodes(self):
        tp_label, tp_pred, fp, fn = np.zeros([4, len(self.label_topology), len(self.prediction_topology)])
        for i, label_pos in self.label_topology.position.items():
            for j, pred_pos in self.prediction_topology.position.items():
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
        label_len, pred_len = len(self.label_topology), len(self.prediction_topology)
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


if __name__ == '__main__':
    cmp_score = ComponentScore('alaki', 'seg_LI_2019-09-19_emb1_pos1_tp264_A4_B4',
                                'alaki', 'one-ten_40_val_pred0.7_LI_2019-09-19_emb1_pos1_tp264_A4_B4')
    cyc_score = CycleScore('alaki', 'seg_LI_2019-09-19_emb1_pos1_tp264_A4_B4',
                                'alaki', 'one-ten_40_val_pred0.7_LI_2019-09-19_emb1_pos1_tp264_A4_B4')
    cmp_score.write_normalized_iou()
    cyc_score.write_normalized_iou()
    print(cmp_score.final_score())
    print(cyc_score.final_score())