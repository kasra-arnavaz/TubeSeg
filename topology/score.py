from abc import ABC, abstractproperty
import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from graph.nx_graph import Component, Cycle, NxGraph
from topology.unpickle import read_pickle
from time_filtering.trackpy_filtering import TrackpyFiltering
from time_filtering.seq_filtering import SequentialFiltering
from time_filtering.seq_filtering_patch import SequentialFilteringPatches

class TopologyScore(ABC):

    def __init__(self, label_path, label_name, pred_path, pred_name, matching_radius=10, zero_thr=0.3, one_thr=0.7):
        self.label_path = label_path
        self.label_name = label_name
        self.pred_path = pred_path
        self.pred_name =  pred_name
        self.matching_radius = matching_radius
        self.zero_thr = zero_thr
        self.one_thr = one_thr
        self.label_pos = self.label_topology.position
        self.pred_pos = self.prediction_topology.position
    
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

    def get_col_row(self):
        patch_map = np.array(['A4','B4','C4','D4',
                    'A3','B3','C3','D3',
                    'A2','B2','C2','D2',
                    'A1','B1','C1','D1'])
        index = int(np.argwhere(patch_map == self.patch_id))
        col, row = np.unravel_index(index, (4,4))
        return col, row

    def get_allowed_range(self):
        col, row = self.get_col_row()
        return [col*256, (col+1)*256-1], [row*256, (row+1)*256-1]

    def match_nodes(self):
        tp_label, tp_pred, fp, fn = np.zeros([4, len(self.label_pos), len(self.pred_pos)])
        for i, label_pos in self.label_pos.items():
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
        label_len, pred_len = len(self.label_pos), len(self.pred_pos)

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


class FilteredCycleScore(TopologyScore):

    @property
    def label_topology(self):
        return read_pickle(f'{self.label_path}/{self.label_name}.cyc')

    @property
    def prediction_topology(self):
        return read_pickle(f'{self.pred_path}/{self.pred_name}.cyctpy')

    @property
    def topology_type(self):
        return 'cyctpy'

    def rescale_pos(self):
        self.pred_pos = {j: pred_pos for j, pred_pos in self.prediction_topology.position.items() if self.is_in_patch(pred_pos)}
        col, row = self.get_col_row()
        self.label_pos = {}
        for i, label_pos in self.label_topology._position.items():
            label_pos[:,2] = (row)*256 + label_pos[:,2]
            label_pos[:,1] = (col)*256 + label_pos[:,1]
            self.label_pos[i] = label_pos

    def is_in_patch(self, pos):
        y_range, x_range = self.get_allowed_range()
        y, x = pos[:,1], pos[:,2]
        return all((x <= x_range[1]) & (x>=x_range[0]) & (y <= y_range[1]) & (y>=y_range[0])) 
        
    def __init__(self, label_path, label_name, pred_path, pred_name, matching_radius=10, zero_thr=0.3, one_thr=0.7):
        super().__init__(label_path, label_name, pred_path, pred_name, matching_radius, zero_thr, one_thr)
        self.rescale_pos()

class PickledCycleScore(TopologyScore):

    @property
    def label_topology(self):
        return read_pickle(f'{self.label_path}/{self.label_name}.cyc')

    @property
    def prediction_topology(self):
        return read_pickle(f'{self.pred_path}/{self.pred_name}.cyc')

    @property
    def topology_type(self):
        return 'cyc'


class FilteredCmpScore(TopologyScore):

    @property
    def label_topology(self):
        return read_pickle(f'{self.label_path}/{self.label_name}.cmp')

    @property
    def prediction_topology(self):
        return read_pickle(f'{self.pred_path}/{self.pred_name}.{self.topology_type}')

    @property
    def topology_type(self):
        return 'cmpseq5'

    def __init__(self, label_path, label_name, pred_path, pred_name, matching_radius=10, zero_thr=0.3, one_thr=0.7):
        super().__init__(label_path, label_name, pred_path, pred_name, matching_radius, zero_thr, one_thr)
        self.rescale_pos()

    def is_in_patch(self, pos):
        y_range, x_range = self.get_allowed_range()
        y, x = pos[1], pos[2]
        return (x <= x_range[1]) & (x>=x_range[0]) & (y <= y_range[1]) & (y>=y_range[0])

    def is_in_patch_2d(self, pos):
        y_range, x_range = self.get_allowed_range()
        y, x = pos[0], pos[1]
        return (x <= x_range[1]) & (x>=x_range[0]) & (y <= y_range[1]) & (y>=y_range[0])

    def rescale_pos(self):
        self.pred_pos = {}
        for j, pred_pos in self.prediction_topology.position.items():
            nodes = []
            for node in pred_pos:
                if self.is_in_patch(node): nodes.append(node)
            if len(nodes) > 0:
                self.pred_pos[j] = np.array(nodes)
        col, row = self.get_col_row()
        self.label_pos = {}
        for i, label_pos in self.label_topology._position.items():
            label_pos[:,2] = (row)*256 + label_pos[:,2]
            label_pos[:,1] = (col)*256 + label_pos[:,1]
            self.label_pos[i] = label_pos

    def plot_filtered_pred_graph(self):
        G = nx.Graph()
        nodes, position = [], {}
        col, row = self.get_col_row()
        for node, pos in self.prediction_topology.nodes_xy_position.items():
            if self.is_in_patch_2d(pos):
                nodes.append(node)
                position[node] = np.array(pos) - np.array([col*256, row*256])
        # G.add_nodes_from(nodes)
        edges = []
        for i, j in self.prediction_topology.edge_list:
            if (i in nodes) and (j in nodes): edges.append((i, j))
        G.add_edges_from(edges)
        null = nx.Graph()
        null.add_nodes_from([0,1,2,3])
        size = 256
        pos_null={0:[0,0], 1:[0,size], 2:[size,0], 3:[size,size]} 
        plt.title(f'{self.label_name}_#C{len(self.pred_pos)}', y=-0.1)
        nx.draw(null, pos_null, node_size=1e-10)
        nx.draw(G, pos=position, 
                node_size=50, node_color='r', with_labels=False)
        plt.show()
        
class PickledCmpScore(TopologyScore):

    @property
    def label_topology(self):
        return read_pickle(f'{self.label_path}/{self.label_name}.cmp')

    @property
    def prediction_topology(self):
        return read_pickle(f'{self.pred_path}/{self.pred_name}.{self.topology_type}')

    @property
    def topology_type(self):
        return 'cmp'


def mean_set_score(label_path, pred_path, thr, model_name, epoch):
    names = [name.replace('.tif', '').replace('label_', '') for name in os.listdir(f'{label_path}/label') if name.endswith('.tif')]
    cyc_scores, cmp_scores, total_scores, cyc_re, cyc_pr = [], [], [], [], []
    for name in names:
        cyc_score = CycleScore(f'{label_path}/label', f'label_{name}', pred_path, f'pred-{thr}-{model_name}-{epoch}_{name}')
        mean_row_cyc, mean_col_cyc = cyc_score.mean_row_and_column_scores()
        cyc_scores.append(cyc_score.final_score())
        cyc_re.append(mean_row_cyc)
        cyc_pr.append(mean_col_cyc)
        cmp_score = ComponentScore(f'{label_path}/label', f'label_{name}', pred_path, f'pred-{thr}-{model_name}-{epoch}_{name}').final_score()
        cmp_scores.append(cmp_score)
        total_scores.append(0.5*(cyc_score.final_score() + cmp_score))
        print(name, mean_row_cyc, mean_col_cyc)
    cyc_re = np.array(cyc_re)
    cyc_pr = np.array(cyc_pr)
    cyc_scores = np.array(cyc_scores)
    cmp_scores = np.array(cmp_scores)
    total_scores = np.array(total_scores)
    print(f'cyc_re: {np.mean(cyc_re):.3f}±{np.std(cyc_re):.3f}')
    print(f'cyc_pr: {np.mean(cyc_pr):.3f}±{np.std(cyc_pr):.3f}')

    print(f'cyc_score: {np.mean(cyc_scores):.3f}±{np.std(cyc_scores):.3f}')
    print(f'cmp_score: {np.mean(cmp_scores):.3f}±{np.std(cmp_scores):.3f}')
    print(f'total_score: {np.mean(total_scores):.3f}±{np.std(total_scores):.3f}')

def filtered_ts_topo_score(topo_type, path, path_current_frame):

    names = [name.replace(f'.{topo_type}', '') for name in os.listdir(path) if name.endswith(f'.{topo_type}')]
    scores = []
    for label_name in names:
        pred_name = label_name.replace('label', 'pred-0.7-semi-40')
        topo_score = FilteredCmpScore(path , label_name, path_current_frame, pred_name)
        scores.append(topo_score.final_score())
    topo_scores = np.array(scores)
    print(f'topo_score: {np.mean(topo_scores):.3f}±{np.std(topo_scores):.3f}')

