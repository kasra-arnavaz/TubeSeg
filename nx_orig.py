import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tifffile as tif
import os
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.metrics import confusion_matrix, log_loss
from scipy import sparse
from PyGEL3D import gel
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog, linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
##from skimage.morphology import *
from collections import Counter
import warnings
from scipy.stats import entropy
import itertools
from scipy.ndimage.morphology import binary_dilation
import tensorflow as tf


class Graph:

    def __init__(self, name, path, make_new=False, create_cyc_obj=False):
        print(name)
        self.name = name
        self.path = path

        self.make_new = make_new
        self.create_cyc_obj = create_cyc_obj
        self.make_skel()
        self.get_graph()
        
        
    def make_skel(self, crop=3):
        
        if (f'skel_{self.name}.graph' in os.listdir(self.path)) and self.make_new==False:
##            print('Using existing skeleton ...')
            pass
        else:
            print('Making new skeleton ...')
            v = tif.imread(f'{self.path}/{self.name}.tif')[:,crop:-crop, crop:-crop]
            positions = np.argwhere(v).astype(np.float32)
            neigh = NearestNeighbors(n_neighbors=4, radius=1.9, metric='l2')
            neigh.fit(positions)
            radius_neighbors_graph = neigh.radius_neighbors_graph(positions).astype(np.bool)
            i, j, _ = sparse.find(sparse.triu(radius_neighbors_graph, k=1))
            edges = np.asarray([i, j]).transpose()

            with open(f'{self.path}/{self.name}.graph','w') as f:
                for p in positions:
                    print("n", p[0], p[1], p[2], file=f)
                for e in edges:
                    print("c", e[0], e[1], file=f)
            g = gel.graph_load(f'{self.path}/{self.name}.graph')
            gel.graph_edge_contract(g, 3)
            s = gel.graph_LS_skeleton(g)
            gel.graph_prune(s)
            gel.graph_save(f'{self.path}/skel_{self.name}.graph', s)
            with open(f'{self.path}/skel_{self.name}.graph') as f:
                lines = f.read().splitlines()
            with open(f'{self.path}/skel_{self.name}.graph', 'w') as f:
                for line in lines:
                    if line.startswith('n') and 'nan' not in line:
                        print(line.split(' ')[0], float(line.split(' ')[1])*(2.89),float(line.split(' ')[2]), float(line.split(' ')[3]), file=f)
                    else: print(line, file=f)
            s = gel.graph_load(f'{self.path}/skel_{self.name}.graph')
            m = gel.graph_to_mesh_cyl(s, fudge=0.5)
            gel.obj_save(f'{self.path}/skel_{self.name}.obj', m)
            os.remove(f'{self.path}/{self.name}.graph')

    def get_graph(self):
    
        df = pd.read_table(f'{self.path}/skel_{self.name}.graph', sep=' ', header=None, usecols=[0,1,2,3])
        is_nan = df.loc[:,1].astype('str').str.contains('nan')
        nodes = df.loc[(~is_nan) &(df[0]=='n'), 1:3].astype('float')
        edges = df.loc[df[0] == 'c', [1,2]].astype('int')
        elist = list(edges.itertuples(index=False, name=None))
        self.pos2, self.pos3, self.cyc_pos, self.cmp_pos = {}, {}, {}, {}
        self.G = nx.Graph()
        self.G.add_edges_from(elist)
        cmp = nx.algorithms.components.connected_components(self.G)
        self.cyc_cntr, self.cyclist, self.cmp_cntr, self.cmp, self.cyc_cntr3 = [], [], [], [], []
        rm_nodes = []
        for c in cmp:
            if len(c)<5:
                rm_nodes.append(list(c))
            else:
                self.cmp.append(list(c))
        rm_nodes = [i for ls in rm_nodes for i in ls]
        self.G.remove_nodes_from(rm_nodes)
        self.cyc = nx.algorithms.cycles.minimum_cycle_basis(self.G)
        self.n_cyc = len(self.cyc)
        for i, c_list in enumerate(self.cyc):
            self.cyc_cntr.append(nodes.loc[c_list,2:3].mean().tolist())
            self.cyc_cntr3.append(nodes.loc[c_list,1:3].mean().tolist())
            self.cyc_pos[i] = df.iloc[c_list, [1,2,3]].to_numpy(dtype='float')
            for c in c_list:
                self.cyclist.append(c)
        if self.n_cyc>0:
            self.cyc_cntr3 = np.array(self.cyc_cntr3)
            self.cyc_cntr3[:,0] = self.cyc_cntr3[:,0]/2.89
            self.cyc_cntr3 = np.round(self.cyc_cntr3)
        self.n_cmp = len(self.cmp)
        for i, cmp in enumerate(self.cmp):
                self.cmp_cntr.append(nodes.loc[list(cmp), [2,3]].mean().tolist())
                self.cmp_pos[i] = df.iloc[list(cmp), [1,2,3]].to_numpy(dtype='float')
        self.Cntr_cyc = nx.Graph()
        self.Cntr_cmp = nx.Graph()
        self.Cntr_cyc.add_nodes_from(range(self.n_cyc))
        self.Cntr_cmp.add_nodes_from(range(len(self.cmp_pos)))
        self.pos_cyc_cntr, self.pos_cmp_cntr = {}, {}
        for i, c in enumerate(self.cyc_cntr):
            self.pos_cyc_cntr[i] = c
        for i, c in enumerate(self.cmp_cntr):
            self.pos_cmp_cntr[i] = c
        for n in self.G.nodes:
            self.pos3[n] = nodes.loc[n, [1,2,3]].to_list()
            self.pos2[n] = nodes.loc[n, [2,3]].to_list()
            
        if self.create_cyc_obj:
            with open(f'{self.path}/skel_{self.name}.graph') as f:
                lines = f.read().splitlines()
            with open(f'{self.path}/cyc_{self.name}.graph', 'w') as fw:
                cyc_node = []
                for i, line in enumerate(lines):
                    if line.startswith('n'):
                        x1, x2, x3 = float(line.split(' ')[1]), float(line.split(' ')[2]), float(line.split(' ')[3])
                        row = np.array([x1, x2, x3])
                        for val in self.cyc_pos.values():
                            for val_row in val:
                                if (val_row == row).all():
                                   cyc_node.append(str(i))
                for i, line in enumerate(lines):
                    if line.startswith('n'):
                        print(line, file=fw)
                    elif (line.split(' ')[1] in cyc_node) or (line.split(' ')[2] in cyc_node):
                        print(line, file=fw)
            with open(f'{self.path}/notcyc_{self.name}.graph', 'w') as fn:
                cyc_node = []
                for i, line in enumerate(lines):
                    if line.startswith('n'):
                        x1, x2, x3 = float(line.split(' ')[1]), float(line.split(' ')[2]), float(line.split(' ')[3])
                        row = np.array([x1, x2, x3])
                        for val in self.cyc_pos.values():
                            for val_row in val:
                                if (val_row == row).all():
                                   cyc_node.append(str(i))
                for i, line in enumerate(lines):
                    if line.startswith('n'):
                        print(line, file=fn)
                    elif (line.split(' ')[1] not in cyc_node) and (line.split(' ')[2] not in cyc_node):
                        print(line, file=fn)
                    
                                                                      
            sc = gel.graph_load(f'{self.path}/cyc_{self.name}.graph')
            snc = gel.graph_load(f'{self.path}/notcyc_{self.name}.graph')
            mc = gel.graph_to_mesh_cyl(sc, fudge=0.5)
            gel.obj_save(f'{self.path}/cyc_{self.name}.obj', mc)
            mnc = gel.graph_to_mesh_cyl(snc, fudge=0.5)
            gel.obj_save(f'{self.path}/notcyc_{self.name}.obj', mnc)


    @classmethod
    def find_feasible_point(cls, halfspaces):
        
        norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
        c = np.zeros((halfspaces.shape[1],))
        c[-1] = -1
        A = np.hstack((halfspaces[:, :-1], norm_vector))
        b = - halfspaces[:, -1:]
        res = linprog(c, A_ub=A, b_ub=b, bounds = (None,None))
        
        return res.x[:-1], res.status == 2


    @classmethod
    def iou(cls, self, other):

        pos_t = self.cyc_pos
        pos_y = other.cyc_pos
        D = np.zeros([len(pos_t), len(pos_y)])
        for i, pointsP in pos_t.items():
            for j,  pointsQ in pos_y.items():
                hullP = ConvexHull(pointsP)
                halfspacesP = hullP.equations
                hullQ = ConvexHull(pointsQ)
                halfspacesQ = hullQ.equations
                halfspacesPQ = np.vstack([halfspacesP,halfspacesQ])
                x_feas, nofeasible = cls.find_feasible_point(halfspacesPQ)
                feasible = np.all(halfspacesPQ[:,:-1] @ x_feas + halfspacesPQ[:,-1] <= 0)
                #if the problem is infeasible, P and Q are not intersecting, x has no meaning.
                if feasible:
                    PQ_vertices = HalfspaceIntersection(halfspacesPQ, x_feas).intersections
                    volumeP = hullP.volume
                    volumeQ = hullQ.volume
                    volumePQ = ConvexHull(PQ_vertices).volume
                    volumePorQ=volumeP + volumeQ - volumePQ
                    D[i,j] = volumePQ/volumePorQ
                else:
                    D[i,j] = 0
        D_df = pd.DataFrame(np.round(D, 2), index=range(self.n_cyc), columns=range(other.n_cyc))
        pd.DataFrame(D, index=range(D.shape[0]), columns=range(D.shape[1])).to_csv(f'{other.path}/iou_{other.name}.csv')

        return D


    def hausdorff(self, pos_t, pos_y):
        
        D = np.zeros([len(pos_t), len(pos_y)])
        for i, pointsP in pos_t.items():
            for j,  pointsQ in pos_y.items():
                D[i,j] = np.maximum(directed_hausdorff(pointsP, pointsQ)[0],directed_hausdorff(pointsQ, pointsP)[0])
        pd.DataFrame(D, index=range(D.shape[0]), columns=range(D.shape[1])).to_csv(f'{self.path}/hausdorff_{self.name}.csv')

        return D

    @staticmethod
    def euclidean(A, B):
        p1 = np.sum(A**2, 1)[:, np.newaxis]
        p2 = np.sum(B**2, 1)
        p3 = -2*np.dot(A,B.T)
        return np.sqrt(p1+p2+p3)

    
    def chamfer(self, pos_t, pos_y):

        D = np.zeros([len(pos_t), len(pos_y)])
        for i, array1 in pos_t.items():
            for j, array2 in pos_y.items():
                distances = np.zeros([array1.shape[0], array2.shape[0]])
                for k in range(array1.shape[0]):
                    distances[k] = np.sum((array1[k] - array2)**2, 1)
                nearest_neighbor = np.min(distances, axis=1)
                D[i,j] = np.mean(nearest_neighbor)
        pd.DataFrame(D, index=range(D.shape[0]), columns=range(D.shape[1])).to_csv(f'{self.path}/chamfer_{self.name}.csv')
        
        return D


    def matching_by_nodes_iou(self, other, mode, radius=10, plot=False):

        if mode == 'cmp':
            self_n_x, other_n_x = self.n_cmp, other.n_cmp
            self_x_pos, other_x_pos = self.cmp_pos, other.cmp_pos
        elif mode == 'cyc':
            self_n_x, other_n_x = self.n_cyc, other.n_cyc
            self_x_pos, other_x_pos = self.cyc_pos, other.cyc_pos
        
        S = np.zeros([self_n_x, other_n_x])
        if self_n_x > 0 and other_n_x > 0:
            score_row, score_col = [], []        
            for i, self_nodes in self_x_pos.items():
                self_nodes = np.array(self_nodes)
                for j, other_nodes in other_x_pos.items():
                    other_nodes = np.array(other_nodes)
                    D = Graph.euclidean(self_nodes, other_nodes)
                    gt_nearest_neighbor_distance = np.amin(D, 1)
                    pred_nearest_neighbor_distance = np.amin(D,0)
                    tp_gt = np.sum(gt_nearest_neighbor_distance <= radius)
                    tp_pred = np.sum(pred_nearest_neighbor_distance <= radius)
                    fp = D.shape[1] - tp_pred
                    fn = D.shape[0] - tp_gt 
                    iou = (tp_gt+tp_pred)/(tp_gt+tp_pred+2*fp+2*fn)
                    n1 = tp_gt/(tp_gt+fn)
                    n2 = tp_pred/(tp_pred+fp)
                    d1 = (tp_gt+2*fn)/(tp_gt+fn)
                    d2 = (tp_pred+2*fp)/(tp_pred+fp)
                    iou_normalized = (n1+n2)/(d1+d2)
##                    if iou_normalized >= 0.7: iou_normalized = 1
                    S[i,j] = iou_normalized
##            S_filtered = S*(S>=0.3)
##            score_row = np.minimum(1, np.sum(S_filtered, 1))
##            score_col = np.minimum(1, np.sum(S_filtered, 0))
            score_row = np.minimum(1, np.sum(S, 1))
            score_col = np.minimum(1, np.sum(S, 0))
       
        elif self_n_x ==0 and other_n_x==0:
            score_row, score_col = [1], [1]
        elif self_n_x == 0 and other_n_x >0:
            score_row, score_col = [1/(1+other_n_x)], [1/(1+other_n_x)]
        elif self_n_x > 0 and other_n_x == 0:
            score_row, score_col = [1/(1+self_n_x)], [1/(1+self_n_x)]
            
        score_final = (np.mean(score_row) + np.mean(score_col))/2
        S_df = pd.DataFrame(S, index=range(self_n_x), columns=range(other_n_x))
        S_df.to_csv(f'{other.path}/{mode}_{other.name}.csv')
        if plot:
            plt.figure(figsize=(20,10))
            plt.suptitle(f'{S_df} \n score_row:{score_row}, score_col:{score_col}, score_final:{score_final}')
            plt.subplot(121)
            self.plot_one_graph(mode=mode)
            plt.subplot(122)
            other.plot_one_graph(mode=mode)
            plt.savefig(f'{other.path}/{mode}_{other.name}.png')
            plt.savefig(f'{other.path}/{mode}_{other.name}.pdf')
            plt.close()

        return S_df, score_row, score_col, score_final


        

    def my_matching(self, other):
    
        s_cmp, sr_cmp, sc_cmp, sf_cmp = Graph.matching_by_nodes_iou(self, other, mode='cmp')
        s_cyc, sr_cyc, sc_cyc, sf_cyc = Graph.matching_by_nodes_iou(self, other, mode='cyc')
##        os.makedirs(f"{other.path}/score_and_entropy", exist_ok=True)
##        np.save(f"{other.path}/score_and_entropy/sc_cmp_{other.name}.npy", sc_cmp)
##        np.save(f"{other.path}/score_and_entropy/sc_cyc_{other.name}.npy", sc_cyc)
##        np.save(f"{other.path}/score_and_entropy/cmp_ent_{other.name}.npy", other.ent_cmp)
##        np.save(f"{other.path}/score_and_entropy/cyc_ent_{other.name}.npy", other.ent_cyc)
        S_cmp = np.stack((np.mean(sr_cmp), np.mean(sc_cmp), sf_cmp), axis=0)
        S_cyc = np.stack((np.mean(sr_cyc), np.mean(sc_cyc), sf_cyc), axis=0)
##        ss = [[sr_cmp, sc_cmp],[sr_cyc, sc_cyc]]
##        SS = pd.DataFrame(ss, index=['cmp', 'cyc'], columns=['row', 'col'])
##        plt.figure(figsize=(20,10))
##        plt.suptitle(f'cyc ent: {np.round(other.ent_cyc,1)}\n \
##                        cyc sco: {np.round(sc_cyc,1)}\n \
##                        cmp ent: {np.round(other.ent_cmp,1)}\n \
##                        cmp sco: {np.round(sc_cmp,1)}')
##        plt.suptitle(f'final score cyc:{sf_cyc:.2f}')
##        plt.suptitle(f'{sr_cyc}\n{sc_cyc}')
##        plt.subplot(121)
##        self.plot_one_graph(mode='both')
##        plt.subplot(122)
##        other.plot_one_graph(mode='both')
##        plt.savefig(f'{other.path}/{other.name}.png')
##        plt.savefig(f'{other.path}/{other.name}.pdf')
##        plt.close()

        return S_cmp, S_cyc



    @classmethod
    def confusion(cls, D_matrix, D_name, D_type, threshold):

        if D_name == 'iou':
            row, col = linear_sum_assignment(-D_matrix)
            mask = D_matrix[row,col]>threshold
            cls.matches_cyc = np.array([row[mask], col[mask]]).T
            tp = cls.matches_cyc.shape[0]
        elif D_name == 'hausdorff' or 'chamfer':
            row, col = linear_sum_assignment(D_matrix)
            mask = D_matrix[row,col]<threshold
            cls.matches_cmp = np.array([row[mask], col[mask]]).T
            tp = cls.matches_cmp.shape[0]
        else: raise Exception(f'D_name {D_name} not defined!')        
        fp = D_matrix.shape[1] - tp
        fn = D_matrix.shape[0] - tp

        return tp, fp, fn


    def plot_one_graph(self, mode, save=False):


        null = nx.Graph()
        null.add_nodes_from([0,1,2,3])
        pos_null={0:[0,0], 1:[0,255], 2:[255,0], 3:[255,255]}
  
        plt.title(f'{self.name}_#L{self.n_cyc}_#C{self.n_cmp}', y=0)
        nx.draw(null, pos=pos_null, node_color='w', node_size=1e-10)
        nx.draw(self.G, pos=self.pos2, node_size=50, node_color='r', with_labels=False, font_size=8)
        if (mode == 'cyc') or (mode == 'both'):
            nx.draw(self.G, pos=self.pos2, node_size=50, node_color='g', with_labels=False, font_size=8, nodelist=self.cyclist)
            nx.draw(self.Cntr_cyc, pos=self.pos_cyc_cntr, node_size=1e-10, node_color='w', node_shape='s', with_labels=True, font_color='k', font_size=12)
        if (mode == 'cmp') or (mode == 'both'):
            nx.draw(self.Cntr_cmp, pos=self.pos_cmp_cntr, node_size=100, node_color='k', node_shape='s', with_labels=True, font_color='w')

        if save:
            plt.savefig(f'{self.path}/skel_{self.name}.png')
            plt.savefig(f'{self.path}/skel_{self.name}.pdf')

        
        return plt.gcf()

    @property    
    def ent_cyc(self):
        os.makedirs(f'{self.path}/sur/', exist_ok=True)
        temp = self.name.split('_')
        temp[3] = 'ent'
        ent_name = '_'.join(temp)
        ent = tif.imread(f'{self.path.replace("preds","ents")}/ent/{ent_name}.tif')
        ones = np.ones_like(ent)
        all_pos = np.argwhere(ones)
        all_pos[:,0] = 2.89*all_pos[:,0]
        ent_pos = np.argwhere(ent)
        mask_ent = np.zeros((self.n_cyc,)+ent.shape, dtype=bool)
        ent_cyc = np.zeros(self.n_cyc)
        for cyc_num in range(self.n_cyc):
            cyc_pos = self.cyc_pos[cyc_num]
            cyc_nodes = self.cyc[cyc_num]
            s = set(cyc_nodes)
            cyc_nodes = np.array(cyc_nodes)
            for pair in itertools.combinations(s, 2):
                if self.G.has_edge(pair[0], pair[1]):
                    mask_nodes = (cyc_nodes == pair[0]) + (cyc_nodes == pair[1])
                    p, q = cyc_pos[mask_nodes]
                    t = np.linspace(0,1,10).reshape(-1,1)
                    p, q = p.reshape(1,-1), q.reshape(1,-1)
                    points_on_edge = (t@p) + ((1-t)@q)
                    points_on_edge = np.array(points_on_edge)
                    distance = self.euclidean(all_pos, points_on_edge)
                    mask_radius = distance<5
                    mask_ent[cyc_num] += np.sum(mask_radius, axis=1, dtype=bool).reshape(ent.shape)
            ent_cyc[cyc_num] = ent[mask_ent[cyc_num]].mean()
        mask_ent_all = np.sum(mask_ent, axis=0, dtype=bool)
        tif.imwrite(f'{self.path}/sur/cycsur_{self.name}.tif', (mask_ent_all*255).astype('uint8'))
        return ent_cyc

    @property    
    def ent_cmp(self):
        os.makedirs(f'{self.path}/sur/', exist_ok=True)
        temp = self.name.split('_')
        temp[3] = 'ent'
        ent_name = '_'.join(temp)
        ent = tif.imread(f'{self.path.replace("preds","ents")}/ent/{ent_name}.tif')
        ones = np.ones_like(ent)
        all_pos = np.argwhere(ones)
        all_pos[:,0] = 2.89*all_pos[:,0]
        ent_pos = np.argwhere(ent)
        mask_ent = np.zeros((self.n_cmp,)+ent.shape, dtype=bool)
        ent_cmp = np.zeros(self.n_cmp)
        for cmp_num in range(self.n_cmp):
            cmp_pos = self.cmp_pos[cmp_num]
            cmp_nodes = self.cmp[cmp_num]
            s = set(cmp_nodes)
            cmp_nodes = np.array(cmp_nodes)
            for pair in itertools.combinations(s, 2):
                if self.G.has_edge(pair[0], pair[1]):
                    mask_nodes = (cmp_nodes == pair[0]) + (cmp_nodes == pair[1])
                    p, q = cmp_pos[mask_nodes]
                    t = np.linspace(0,1,10).reshape(-1,1)
                    p, q = p.reshape(1,-1), q.reshape(1,-1)
                    points_on_edge = (t@p) + ((1-t)@q)
                    points_on_edge = np.array(points_on_edge)
                    distance = self.euclidean(all_pos, points_on_edge)
                    mask_radius = distance<5
                    mask_ent[cmp_num] += np.sum(mask_radius, axis=1, dtype=bool).reshape(ent.shape)
            ent_cmp[cmp_num] = ent[mask_ent[cmp_num]].mean()
        mask_ent_all = np.sum(mask_ent, axis=0, dtype=bool)
        tif.imwrite(f'{self.path}/sur/cmpsur_{self.name}.tif', (mask_ent_all*255).astype('uint8'))
        return ent_cmp

                

        
    def compare_matching(self, other, matches_cyc, matches_cmp):

        null = nx.Graph()
        null.add_nodes_from([0,1,2,3])
        pos_null={0:[0,0], 1:[0,255], 2:[255,0], 3:[255,255]}
        

        fig = plt.figure(figsize=(20,10))
        fig.suptitle(f'tp_cyc: {matches_cyc.shape[0]}, tp_cmp:{matches_cmp.shape[0]}')
        plt.subplot(121)
        plt.title(f'{self.name}_#L{self.n_cyc}_#C{self.n_cmp}')
        nx.draw(null, pos=pos_null, node_color='w', node_size=1e-10)
        nx.draw(self.G, pos=self.pos2, node_size=50, node_color='k', with_labels=False, font_size=8)
        for i, c in enumerate(matches_cmp):
            nx.draw(self.G, pos=self.pos2, node_size=50, node_color=f'C{i}', with_labels=False, font_size=8, nodelist=self.cmp[c[0]])
        for i, c in enumerate(matches_cyc):
            nx.draw(self.G, pos=self.pos2, nodelist=[], edge_color=f'C{9-i}', width=5,
                    edgelist=[e for e in self.G.edges() if e[0] and e[1] in self.cyc[c[0]]])
        plt.subplot(122)
        plt.title(f'{other.name}_#L{other.n_cyc}_#C{other.n_cmp}')
        nx.draw(null, pos=pos_null, node_color='w', node_size=1e-10)
        nx.draw(other.G, pos=other.pos2, node_size=50, node_color='k', with_labels=False, font_size=8)
        for i, c in enumerate(matches_cmp):
            nx.draw(other.G, pos=other.pos2, node_size=50, node_color=f'C{i}', with_labels=False, font_size=8, nodelist=other.cmp[c[1]])
        for i, c in enumerate(matches_cyc):
            nx.draw(other.G, pos=other.pos2, nodelist=[], edge_color=f'C{9-i}', width=5,
                    edgelist=[e for e in other.G.edges() if e[0] and e[1] in other.cyc[c[1]]])
        plt.savefig(f'{other.path}/{other.name}.png')
        plt.savefig(f'{other.path}/{other.name}.pdf')
        plt.close()

def make_patches(target_path, pred_path, prefix, split, epoch, thr):

    names_target = [f.replace('seg_','') for f in os.listdir(target_path) if f.endswith('tif') ]
    tags = ['A4','B4','C4','D4',
           'A3','B3','C3','D3',
           'A2','B2','C2','D2',
           'A1','B1','C1','D1']
    for name in names_target:
        patch_id = name.split('_')[-1][:2]
        raw_name = name.replace('_'+patch_id, '')
        t = tif.imread(f'{target_path}/seg_{name}').reshape(-1)
##        y = tif.imread(f'{pred_path}/{prefix}_{epoch}_{split}_pred{np.round(thr, 2)}_{raw_name}')
        y = tif.imread(f'{pred_path}/{prefix}_{epoch}_{split}_lin_{raw_name}')
        y = y.reshape(-1,4,256,4,256).transpose(0,1,3,2,4)
        y = y.reshape(-1,16,256,256).transpose(1,0,2,3,)
        for num, let in enumerate(tags):
            if let == patch_id:
                patch_num = num
        y = y[patch_num]
##        saving_path  = pred_path.replace('preds', 'preds_patches')
        saving_path  = pred_path.replace('lins', 'lins_patches')
        os.makedirs(saving_path, exist_ok=True)
        tif.imwrite(f'{saving_path}/{prefix}_{epoch}_{split}_lin_{name}', y, 'minisblack')
##        tif.imwrite(f'{saving_path}/{prefix}_{epoch}_{split}_pred{np.round(thr, 2)}_{name}', y, 'minisblack')
        
def make_4x4(path):

    names = [f.replace('.tif', '') for f in os.listdir(path) if f.endswith('tif')]
    for name in names:
        x = tif.imread(f'{path}/{name}.tif')
        x = x.reshape(-1,4,32,4,32).transpose(0,1,3,2,4)
        x = x.reshape(-1,16,32,32).transpose(1,0,2,3)
        for i in range(16):
            tif.imwrite(f'{path}/{name}_p{i+1}.tif', x[i], 'minisblack')

def put_folder(path,prefix, epoch, split, thr_list, target_path='target_val_patches'):

    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
    for name in names:
        os.makedirs(f'{path}/{name}', exist_ok=True)
        for i in thr_list:
            os.rename(f'{path}/{prefix}_{epoch}_{split}_pred{np.round(i, 2)}_{name}.png', f'{path}/{name}/{prefix}_{epoch}_{split}_pred{np.round(i, 2)}_{name}.png')
            os.rename(f'{path}/{prefix}_{epoch}_{split}_pred{np.round(i, 2)}_{name}.pdf', f'{path}/{name}/{prefix}_{epoch}_{split}_pred{np.round(i, 2)}_{name}.pdf')
            os.rename(f'{path}/cyc_{prefix}_{epoch}_{split}_pred{np.round(i, 2)}_{name}.csv', f'{path}/{name}/cyc_{prefix}_{epoch}_{split}_pred{np.round(i, 2)}_{name}.csv')
            os.rename(f'{path}/cmp_{prefix}_{epoch}_{split}_pred{np.round(i, 2)}_{name}.csv', f'{path}/{name}/cmp_{prefix}_{epoch}_{split}_pred{np.round(i, 2)}_{name}.csv')
##            os.rename(f'{path}/iou_{prefix}_{i+1}_{name}.csv', f'{path}/{name}/iou_{prefix}_{i+1}_{name}.csv')
##            os.rename(f'{path}/chamfer_{prefix}_{i+1}_{name}.csv', f'{path}/{name}/chamfer_{prefix}_{i+1}_{name}.csv')
##            os.rename(f'{path}/hausdorff_{prefix}_{i+1}_{name}.csv', f'{path}/{name}/hausdorff_{prefix}_{i+1}_{name}.csv')

def weighted_sig(path, a, b):
    names = [f for f in os.listdir(path) if 'lin' in f]
    for name in names:
        x = tif.imread(f'{path}/{name}')
        z = a*x + b
        p = 1/(1+np.exp(-z))
        new_name = name.replace('lin', 'prob')
        tif.imwrite(f'{path}/{new_name}', p)
        
def early_stopping(target_path, pred_path, prefix, max_epoch):
   
    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
    conf_cyc = np.zeros([3, len(names), max_epoch]) #tp, fp, fn
    conf_cmp = np.zeros([3, len(names), max_epoch])
    for j in range(max_epoch):
        print(j+1)
        for i, name in enumerate(names):
            t = Graph(f'seg_{name}', target_path)
            y = Graph(f'{prefix}_{j+1}_{name}', pred_path)
   
            D_cyc = Graph.iou(t.cyc_pos, y.cyc_pos)
##            D_cmp = y.hausdorff(t.cmp_pos, y.cmp_pos)
            D_cmp = y.chamfer(t.cmp_pos, y.cmp_pos)
            conf_cyc[0,i,j], conf_cyc[1,i,j], conf_cyc[2,i,j] = Graph.confusion(D_cyc, 'iou', 'cyc', threshold=0.5)
            conf_cmp[0,i,j], conf_cmp[1,i,j], conf_cmp[2,i,j] = Graph.confusion(D_cmp, 'chamfer', 'cmp', threshold=35)
            Graph.compare_matching(t, y, Graph.matches_cyc, Graph.matches_cmp)
    sum_conf_cyc = np.sum(conf_cyc, 1)
    sum_conf_cmp = np.sum(conf_cmp, 1)
    for i, j in enumerate(['tp', 'fp', 'fn']):  
        np.save(f'{pred_path}/{j}_cyc.npy', sum_conf_cyc[i])
        np.save(f'{pred_path}/{j}_cmp.npy', sum_conf_cmp[i])
    f1_cyc = 2*sum_conf_cyc[0]/(2*sum_conf_cyc[0] + sum_conf_cyc[1] + sum_conf_cyc[2])
    f1_cmp = 2*sum_conf_cmp[0]/(2*sum_conf_cmp[0] + sum_conf_cmp[1] + sum_conf_cmp[2])
    f1 = (f1_cyc + f1_cmp)/2
    print(f1)
    plt.figure()
    x = np.arange(1,max_epoch+1)
    plt.plot(x, f1_cyc, label='F1 score loop')
    plt.plot(x, f1_cmp, label='F1 score component')
    plt.plot(x, f1, label='Average F1 score')
    plt.legend()
    plt.xlabel('# epochs')
    plt.savefig(f'{pred_path}/f1.png')
    plt.savefig(f'{pred_path}/f1.pdf')

def new_early_stopping(target_path, pred_path, prefix, epoch_range):
    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
    n_epochs = epoch_range[1]-epoch_range[0]+1
    S = np.zeros([2, 3, len(names), n_epochs]) ## /cmp,cyc/row,col,final
    for j in range(n_epochs):
        for i, name in enumerate(names):
            t = Graph(f'seg_{name}', target_path, make_new=True)
            y = Graph(f'{prefix}_{j+epoch_range[0]}_{name}', pred_path, make_new=True)
            S[0,:,i,j], S[1,:,i,j] = Graph.my_matching(t,y)
##    np.save(f'{pred_path}/score.npy', S)
##    plt.figure()
##    plt.plot(np.mean(S[0,2], 0), label='final score cmp')
##    plt.plot(np.mean(S[1,2], 0), label='final score cyc')
##    plt.plot(np.mean(S[0,0], 0), label='row score cmp')
##    plt.plot(np.mean(S[1,0], 0), label='row score cyc')
##    plt.plot(np.mean(S[0,1], 0), label='column score cmp')
##    plt.plot(np.mean(S[1,1], 0), label='column score cyc')
##    plt.legend()
##    plt.xlabel('# epochs')
##    plt.savefig(f'{pred_path}/score_plot.png')
##    plt.savefig(f'{pred_path}/score_plot.pdf')

def threshold_prob_map(path, thr_list):
    
    saving_path = path.replace('probs', 'preds')
    os.makedirs(saving_path, exist_ok=True)
    names = [f.replace('.tif', '') for f in os.listdir(path) if f.endswith('.tif')]
    for name in names:
        for thr in thr_list:
            x = tif.imread(f'{path}/{name}.tif')
            saving_name = name.replace('prob', f'pred-{np.round(thr, 2)}')
            tif.imwrite(f'{saving_path}/{saving_name}.tif', ((x>thr)*255).astype('uint8'))

def threshold_ent(path, thr_list):
    
    names = [f.replace('.tif', '') for f in os.listdir(path) if f.endswith('.tif')]

    for name in names:
        for thr in thr_list:
            thr = np.round(thr, 2)
            saving_path = f'{path}/ent{thr}'
            os.makedirs(saving_path, exist_ok=True)
            x = tif.imread(f'{path}/{name}.tif')
            saving_name = name.replace('ent', f'ent{thr}')
            tif.imwrite(f'{saving_path}/{saving_name}.tif', ((x>thr)*255).astype('uint8'))
        

def thr_selection(target_path, pred_path, prefix, epoch_list, thr_list, load=False):

    if not load:
        names = [f.replace('.tif', '').replace('label_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
        S = np.zeros([2, 3, len(names), len(thr_list), len(epoch_list)]) ## /cmp,cyc/row,col,final
        for k, ep in enumerate(epoch_list):
            for j, thr in enumerate(thr_list):
                for i, name in enumerate(names):
                    t = Graph(f'label_{name}', target_path, make_new=False)
                    # y = Graph(f'{prefix}_{ep}_{split}_pred{np.round(thr, 2)}_{name}', pred_path, make_new=False)
                    y = Graph(f'pred-{thr}-{prefix}-{ep}_{name}', f'{pred_path}/pred-{thr}/patches', make_new=False)
                    S[0,:,i,j,k], S[1,:,i,j,k] = Graph.my_matching(t,y)                   
        np.save(f'{pred_path}/score_{prefix}_{thr_list}.npy', S)
    S = np.load(f'{pred_path}/score_{prefix}_{thr_list}.npy')
    overall_score = (S[0,2]+S[1,2])/2
    mean_iou = np.mean(overall_score, 0)
    std_iou = np.std(overall_score, 0)
    best_thr_idx, best_epoch_idx =  np.unravel_index(np.argmax(mean_iou), mean_iou.shape)
    best_thr, best_epoch = thr_list[best_thr_idx], epoch_list[best_epoch_idx]
    print(f'Best thr {best_thr} and best epoch {best_epoch} for {prefix}')
    print(f'Best score for {prefix}: {np.round(mean_iou[best_thr_idx,best_epoch_idx], 3)}±{np.round(std_iou[best_thr_idx,best_epoch_idx],3)}')
    if len(thr_list)>1:
        plt.figure()
        plt.plot(thr_list, mean_iou.reshape(-1), label='overall score')
        plt.plot(thr_list, np.mean(S[0,2], 0).reshape(-1), label='final score cmp')
        plt.plot(thr_list, np.mean(S[1,2], 0).reshape(-1), label='final score cyc')
        plt.plot(thr_list, np.mean(S[0,0], 0).reshape(-1), label='row score cmp')
        plt.plot(thr_list, np.mean(S[1,0], 0).reshape(-1), label='row score cyc')
        plt.plot(thr_list, np.mean(S[0,1], 0).reshape(-1), label='column score cmp')
        plt.plot(thr_list, np.mean(S[1,1], 0).reshape(-1) ,label='column score cyc')
        plt.legend()
        plt.xlabel('thresholds')
        plt.savefig(f'{pred_path}/score_plot_thr.png')
        plt.savefig(f'{pred_path}/score_plot_thr.pdf')
    elif len(epoch_list)>1:
        plt.figure()
        plt.plot(epoch_list, mean_iou.reshape(-1), label='overall score')
        plt.plot(epoch_list, np.mean(S[0,2], 0).reshape(-1), label='final score cmp')
        plt.plot(epoch_list, np.mean(S[1,2], 0).reshape(-1), label='final score cyc')
        plt.plot(epoch_list, np.mean(S[0,0], 0).reshape(-1), label='row score cmp')
        plt.plot(epoch_list, np.mean(S[1,0], 0).reshape(-1), label='row score cyc')
        plt.plot(epoch_list, np.mean(S[0,1], 0).reshape(-1), label='column score cmp')
        plt.plot(epoch_list, np.mean(S[1,1], 0).reshape(-1),label='column score cyc')
        plt.legend()
        plt.xlabel('epochs')
        plt.savefig(f'{pred_path}/score_plot_epochs.png')
        plt.savefig(f'{pred_path}/score_plot_epochs.pdf')
    

def thr_selection_pixel(target_path, pred_path, prefix, epoch, split, thr_list):
    names = [f.replace('.tif', '').replace(f'label_{split}_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
    iou = np.zeros([len(thr_list), len(names)])
    for j, thr in enumerate(thr_list):
        for i, name in enumerate(names):
            T, Y = 0, 0
            t = tif.imread(f'{target_path}/label_{split}_{name}.tif').reshape(-1)
            y = tif.imread(f'{pred_path}/pred-{thr}/patches/pred-{np.round(thr, 2)}-{prefix}-{epoch}_{split}_{name}.tif').reshape(-1)
            T = np.append(T,t)
            Y = np.append(Y,y)
            T = np.clip(T[1:], 0, 1)
            Y = np.clip(Y[1:], 0, 1)
        plt.hist(Y, bins=20)
        plt.show()

    #         tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    #         iou[j,i] = tp/(tp+fp+fn)
    # mean_iou = np.mean(iou, 1)
    # std_iou = np.std(iou, 1)
    # idx_best = np.argmax(mean_iou)
    # best_thr = thr_list[idx_best]
    # print(f'Best thr for {prefix}: {best_thr}')
    # print(f'Best score for {prefix}: {mean_iou[idx_best]}±{std_iou[idx_best]}')
    # plt.figure()
    # plt.plot(thr_list, mean_iou, '-o', label='mean')
    # plt.fill_between(thr_list, mean_iou-std_iou, mean_iou+std_iou, alpha=0.5, label='1 std')
    # plt.xlabel('thresholds')
    # plt.ylabel('IoU')
    # plt.legend()
    # plt.savefig(f'{pred_path}/pixel_threshold.png')
    # plt.savefig(f'{pred_path}/pixel_threshold.pdf')


def print_confusion(path):
    for i in ['tp_cyc', 'fp_cyc', 'fn_cyc', 'tp_cmp', 'fp_cmp', 'fn_cmp']:
        print(i, np.load(f'{path}/{i}.npy'))

def plot_recall_precision(path, max_epoch):
    C = np.zeros([6, max_epoch])
    for i, m in enumerate(['tp_cyc', 'fp_cyc', 'fn_cyc', 'tp_cmp', 'fp_cmp', 'fn_cmp']):
        C[i] = np.load(f'{path}/{m}.npy')
    rec_cyc = C[0]/(C[0]+C[2])
    pre_cyc = C[0]/(C[0]+C[1])
    rec_cmp = C[3]/(C[3]+C[5])
    pre_cmp = C[3]/(C[3]+C[4])
    plt.plot(rec_cyc, label='rec_cyc')
    plt.plot(pre_cyc, label='pre_cyc')
    plt.plot(rec_cmp, label='rec_cmp')
    plt.plot(pre_cmp, label='pre_cmp')
    plt.legend()
    plt.savefig(f'{path}/rec_pre.png')
    plt.savefig(f'{path}/rec_pre.pdf')

    

def pixel_early_stopping(target_path, pred_path, prefix ,max_epochs):

    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
    RECALL, PRECISION, F1 = [], [], []
    for j in range(max_epochs):
        T, Y = 0, 0
        for i, name in enumerate(names):
            t = tif.imread(f'{target_path}/seg_{name}.tif').reshape(-1)
            y = tif.imread(f'{pred_path}/{prefix}_{j+1}_{name}.tif').reshape(-1)
            T = np.append(T,t)
            Y = np.append(Y,y)
        T = np.clip(T[1:], 0, 1)
        Y = np.clip(Y[1:], 0, 1)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        accu = (tp+tn)/(tp+tn+fp+fn)
        tpr = tp/(tp+fn)    # hit rate, recall, sensitivity
        tnr = tn/(tn+fp)    # specificity, selectivity
        precision = tp/(tp+fp)
        bal_accu = (tpr + tnr)/2
        f1 = (2*tp)/(2*tp+fp+fn)
        RECALL.append(tpr)
        PRECISION.append(precision)
        F1.append(f1)
    plt.figure()
    plt.plot(RECALL, label='Recall')
    plt.plot(PRECISION, label='Precision')
    plt.plot(F1, label='F1')
    plt.legend()
    plt.savefig(f'{pred_path}/pixel_early_stopping.png')
    plt.savefig(f'{pred_path}/pixel_early_stopping.pdf')

def add_prefix(path, prefix):
    names = [f for f in os.listdir(path) if not f.startswith(prefix)]
    for name in names:
        os.rename(f'{path}/{name}', f'{path}/{prefix}_{name}')

def make_skels_path(path):
    names = [f.replace('.tif', '') for f in os.listdir(path) if f.endswith('.tif')]
    for name in names:
        y = Graph(name, path, make_new=False)

def remove_s(path):
    names = os.listdir(path)

    for name in names:
        if '_s' in name:
            new_name = name.replace('_s', '_')
            os.rename(f'{path}/{name}', f"{path}/{new_name}")

##def mean_ent_(path):
##    names = [f for f in os.listdir(path) if f.endswith('tif')]
##    for name in names:
##        pi = tif.imread(f'{path}/{name}')
##        p = np.stack((pi, 1-pi), axis=0)
##        mask = p > 0.00001
##        h = np.zeros_like(p)
##        h[mask] = np.log2(1 / p[mask])
##        H = np.sum(p*h, axis=0)
##        print(f'{name}: Entropy {np.mean(H)}')

def mean_ent(path):
    names = [f for f in os.listdir(path) if f.endswith('tif')]
    mean_ent_path = []
    for name in names:
        pi = tif.imread(f'{path}/{name}')
        p = np.stack((pi, 1-pi), axis=0)
        H = entropy(p, base=2, axis=0)
        mean_ent_path.append(np.mean(H))
##        print(f'{name}: Entropy {np.mean(H)}')
        return H

def prob2ent(path):
    names = [f for f in os.listdir(path) if f.endswith('tif')]
    ent_path = f'{path}/ent'
    os.makedirs(ent_path, exist_ok=True)
    for name in names:
        pi = tif.imread(f'{path}/{name}')
        p = np.stack((pi, 1-pi), axis=0)
        H = entropy(p, base=2, axis=0)
        tif.imwrite(f'{ent_path}/{name.replace("prob","ent")}', H)
        

##def prob2ent_(path):
##    names = [f for f in os.listdir(path) if f.endswith('tif')]
##    os.makedirs(f'{path}/ent', exist_ok=True)
##    for name in names:
##        pi = tif.imread(f'{path}/{name}')
##        p = np.stack((pi, 1-pi), axis=0)
##        mask = p > 0.00001
##        h = np.zeros_like(p)
##        h[mask] = np.log2(1 / p[mask])
##        H = np.sum(p*h, axis=0)
##        tif.imwrite(f'{path}/ent/{name.replace("prob","ent")}', H)


def tr_loss(path, epoch_list):
    loss = []
    for epoch in epoch_list:
        loss.append(np.load(f'{path}/{epoch}_ae_seg_loss.npy'))
    plt.plot(epoch_list, loss, label='segmentation loss')
    plt.legend()
    plt.ylim((0, 0.15))
    plt.xlabel('# epochs')
    plt.ylabel('Training loss')
    plt.savefig(f'{path}/tr_loss.png')

def tr_loss_m3(path, epoch_list):
    loss, loss_rec, loss_seg = [], [], []
    for epoch in epoch_list:
        loss.append(np.load(f'{path}/one-ten_{epoch}_loss.npy'))
        loss_seg.append(np.load(f'{path}/one-ten_{epoch}_seg_loss.npy'))
        loss_rec.append(np.load(f'{path}/one-ten_{epoch}_rec_loss.npy'))
    plt.plot(epoch_list, loss, label='combined loss')
##    plt.plot(epoch_list, loss_seg, label='segmentation loss')
##    plt.plot(epoch_list, loss_rec, label='reconstruction loss')
##    plt.ylim((0, 0.15))
    plt.legend()
    plt.xlabel('# epochs')
    plt.ylabel('Training loss')
    plt.savefig(f'{path}/tr_loss.png')
    
def select_random(path):
    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(path) if f.endswith('.tif')]
    random_names = np.random.choice(names, size=68, replace=False)
    for name in random_names:
        t = Graph(f'seg_{name}', path)
        print(t.name)
        t.plot_one_graph(mode='both')
        plt.show()

def calibration(target_path, before_path, after_path, prefix, epoch, split, save_to):

    os.makedirs(save_to, exist_ok=True)
    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
    prob_before, prob_after, target, obs_freq_all_before, obs_freq_all_after = [], [], [], [], []
    bins = np.arange(-0.05,1.06,0.1)
    mid_bins = np.arange(0,1.01, 0.1)
    for name in names:
        prob_patch_before = tif.imread(f'{before_path}/{prefix}_{epoch}_{split}_prob_{name}.tif').reshape(-1)
        prob_patch_after = tif.imread(f'{after_path}/{prefix}_{epoch}_{split}_prob_{name}.tif').reshape(-1)
        target_patch = tif.imread(f'{target_path}/seg_{name}.tif').reshape(-1)
        target_patch = np.clip(target_patch, 0, 1)
        prob_before = np.append(prob_before, prob_patch_before)
        prob_after = np.append(prob_after, prob_patch_after)
        target = np.append(target, target_patch)
        obs_freq_patch_before, obs_freq_patch_after = [], []
        for i in range(len(bins)-1):
            mask_before = np.bitwise_and(bins[i]<=prob_patch_before, prob_patch_before<bins[i+1])
            mask_after = np.bitwise_and(bins[i]<=prob_patch_after, prob_patch_after<bins[i+1])
            if np.sum(mask_before) == 0: obs_freq_patch_before.append(0)
            else: obs_freq_patch_before.append(np.mean(target_patch[mask_before]))
            if np.sum(mask_after) == 0: obs_freq_patch_after.append(0)
            else: obs_freq_patch_after.append(np.mean(target_patch[mask_after]))

        plt.figure()
        plt.plot(mid_bins, obs_freq_patch_before, '-o', label='Before')
        plt.plot(mid_bins, obs_freq_patch_after, '-o', label='After')
        plt.plot([0,1], [0,1], 'k-')
        plt.xlabel('Model Probability')
        plt.ylabel('Observed Frequency')
        plt.xticks(mid_bins)
        plt.yticks(mid_bins)
        plt.grid()
        plt.legend()
        plt.title(name)
        plt.savefig(f'{save_to}/cal_{name}.png')
        plt.close()
    for i in range(len(bins)-1):
        mask_before = np.bitwise_and(bins[i]<=prob_before, prob_before<bins[i+1])
        if np.sum(mask_before) == 0:
           obs_freq_all_before.append(0)
        else:
            obs_freq_all_before.append(np.mean(target[mask_before]))
        mask_after = np.bitwise_and(bins[i]<=prob_after, prob_after<bins[i+1])
        if np.sum(mask_after) == 0:
           obs_freq_all_after.append(0)
        else:
            obs_freq_all_after.append(np.mean(target[mask_after]))
    plt.figure()
    plt.plot(mid_bins, obs_freq_all_before, '-o', label='Before')
    plt.plot(mid_bins, obs_freq_all_after, '-o', label='After')
    plt.plot([0,1], [0,1], 'k-')
    plt.xlabel('Model Probability')
    plt.ylabel('Observed Frequency')
    plt.xticks(mid_bins)
    plt.yticks(mid_bins)
    plt.grid()
    plt.legend()
    plt.title('All')
    plt.savefig(f'{save_to}/cal_all.png')
    plt.close()

    
def ent_target_pred(ent_path, target_path, pred_path, thr_list, epoch_list, split, prefix):
    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(target_path) if f.endswith('.tif')]

    for thr in thr_list:
        for epoch in epoch_list:
            for name in names:
                name_pred = f'{prefix}_{epoch}_{split}_pred{thr:.1f}_{name}'
                y = tif.imread(f'{pred_path}/{name_pred}.tif')
                t = tif.imread(f'{target_path}/seg_{name}.tif')
                t = np.clip(t,0,1)
                y = np.clip(y,0,1)
                ent_name = f'{prefix}_{epoch}_{split}_ent_{name}'
                ent = tif.imread(f'{ent_path}/{ent_name}.tif')
                ent = np.stack((ent, ent, ent), axis=-1)
                edge_ent = ent.copy()
                for z in range(y.shape[0]):
                    pred_mask = (y[z] - binary_dilation(y[z]))>0
                    target_mask = (t[z] - binary_dilation(t[z]))>0
                    edge_ent[z,pred_mask,0] = 1.
                    edge_ent[z,target_mask,1] = 1
                    edge_ent[z,pred_mask,1] = 0
                    edge_ent[z,pred_mask,2] = 0
                    edge_ent[z,target_mask,0] = 0
                    edge_ent[z,target_mask,2] = 0
                saving_name = name_pred.replace('pred','edge')
                write_path = f'{ent_path}/../ent_target_pred' 
                os.makedirs(write_path, exist_ok=True)
                tif.imwrite(f'{write_path}/{saving_name}.tif', (edge_ent*255).astype('uint8'), photometric='rgb')

def plot_ent_by_category(target_path, pred_path, ent_path, thr_list, epoch_list, split, prefix):
    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
    write_path = f'{pred_path}/ent_target_pred'
    if 'blurred' in ent_path:
        title_2 = f'with smoothed window {ent_path.split("_")[-1]}'
    else: title_2 = 'non-smooth'
    os.makedirs(write_path, exist_ok=True)
    t, y, ent = 0, 0, 0
    for thr in thr_list:
        for epoch in epoch_list:
            for name in names:
                ent_name = f'{prefix}_{epoch}_{split}_ent_{name}'
                name_pred = f'{prefix}_{epoch}_{split}_pred{thr:.1f}_{name}'
                y_temp = tif.imread(f'{pred_path}/{name_pred}.tif').reshape(-1)
                t_temp = tif.imread(f'{target_path}/seg_{name}.tif').reshape(-1)
                ent_temp = tif.imread(f'{ent_path}/{ent_name}.tif').reshape(-1)
                y = np.append(y, y_temp)
                t = np.append(t, t_temp)
                ent = np.append(ent, ent_temp)
    t = np.clip(t[1:],0,1)
    y = np.clip(y[1:],0,1)
    ent = ent[1:]
    gt_positive = t>0
    pred_positive = y>0
    gt_negative = t==0
    pred_negative = y==0
    TP = gt_positive * pred_positive
    TN = gt_negative * pred_negative
    FP = gt_negative * pred_positive
    FN = gt_positive * pred_negative
    data = [ent[TP], ent[TN], ent[FP], ent[FN]]
    plt.figure()
    bp = plt.boxplot(data, showfliers=False, showmeans=True, meanline=True)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.xticks([1,2,3,4],['TP', 'TN', 'FP', 'FN'])
    plt.ylabel('Entropies of voxels in validation set')
    plt.title(f'{pred_path.split("/")[3].capitalize()} {title_2} at thr {thr_list[0]}')
    plt.savefig(f"{pred_path.replace('pred','ent')}/{pred_path.split('/')[3].capitalize()} {title_2} at thr {thr_list[0]}.png")


def remove_high_entropy_from_prediction(ent_path, ent_thr, pred_thr):
    pred_path = ent_path.replace('ents','preds')
    save_to = ent_path.replace('ents', 'filteredpreds')
    os.makedirs(save_to, exist_ok=True)
    ent_names = [f for f in os.listdir(ent_path) if f.endswith('.tif')]
    for ent_name in ent_names:
        pred_name = ent_name.replace('ent', f'pred{pred_thr:.1f}')
        ent = tif.imread(f'{ent_path}/{ent_name}')
        pred = tif.imread(f'{pred_path}/{pred_name}')
        mask_ent = ent >= ent_thr
        pred[mask_ent] = 0
        saving_name = pred_name.replace(f'pred{pred_thr:.1f}', f'pred{pred_thr:.1f}ent{ent_thr:.1f}')
        tif.imwrite(f'{save_to}/{saving_name}', (pred*255.).astype('uint8'))

def convolve_entropy(ent_path, kernel_size_2d):
    ent_names = [f for f in os.listdir(ent_path) if f.endswith('.tif')]
    save_to = f'{ent_path}/conv_{kernel_size_2d}'
    os.makedirs(save_to, exist_ok=True)
    for ent_name in ent_names:
        ent = tif.imread(f'{ent_path}/{ent_name}')[np.newaxis,[10,11],:,:,np.newaxis]
        print(np.unique(ent))
        ones = tf.ones_like(ent)
        kernel = tf.ones([ent.shape[1]]+kernel_size_2d+[1,1])
        conv_ent = tf.nn.conv3d(ent, kernel, strides=[1,1,1,1,1], padding='SAME')
        conv_ones = tf.nn.conv3d(ones, kernel, strides=[1,1,1,1,1], padding='SAME')
        mean_ent = conv_ent/conv_ones
        with tf.compat.v1.Session() as sess:
            to_write = mean_ent.eval(session=sess)
        tif.imwrite(f'{save_to}/{ent_name}', to_write)
        np.save(f'{save_to}/{ent_name}', to_write)

def plot_hist_ent(path):
    names = [f for f in os.listdir(path) if f.endswith('.tif')]
    hist = []
    for name in names:
        x = tif.imread(f'{path}/{name}')
        hist.append(np.mean(x))
    plt.figure()
    plt.hist(hist)
    plt.xlabel('mean entropy')
    plt.ylabel('# patches')
##    bins=np.arange(0,1.01,0.1)
##    plt.yticks(np.arange(0,len(names)))
    plt.show()


##def get_valid_names():
##    names_val = [f.replace('.tif', '') for f in os.listdir('../Dataset/val/images') if f.endswith('.tif')]
####    names_ts = [f.replace('.tif', '') for f in os.listdir('../Dataset/test/images') if f.endswith('.tif')]
##    return names_val

def get_difficulty():
    x = np.loadtxt('diff_csv.csv', delimiter=',', skiprows=1, usecols=[0,1,3], dtype=np.object)
    x_names = x[:,0]+'_'+x[:,1]
    names_val = [f.replace('.tif', '') for f in os.listdir('../Dataset/val/patches') if f.endswith('.tif')]
##    names_ts = [f.replace('.tif', '') for f in os.listdir('../Dataset/test/patches') if f.endswith('.tif')]
##    names_total = names_val + names_ts
    csv = np.array([x_names, x[:,-1]]).T
    diff, mean_ent = [], []
    for name in names_val:
        idx = np.where(name==x_names)
        diff.append(int(csv[idx, 1]))
        ent = tif.imread(f'm3/val/patches/calibrated/val_ents_patches/ent/one-ten_40_val_ent_{name}.tif')
        mean_ent.append(np.mean(ent))
    print(diff)
    plt.scatter(diff,mean_ent)
    plt.xticks(diff)
    plt.show()

def loop_ent(target_path, pred_path, ent_path, prefix, epoch, split, thr):

    names = [f.replace('.tif', '').replace('seg_', '') for f in os.listdir(target_path) if f.endswith('.tif')]
    S = np.zeros([2, 3, len(names)]) ## /cmp,cyc/row,col,final
    ent_fp, ent_tp, ent_fn = [], [], []
    if 'blurred' in ent_path:
        title_2 = f'with smoothed window {ent_path.split("_")[-1]}'
    else: title_2 = 'non-smooth'
    for i, name in enumerate(names):
        ent_name = f'{prefix}_{epoch}_{split}_ent_{name}'
        t = Graph(f'seg_{name}', target_path, make_new=False)
        y = Graph(f'{prefix}_{epoch}_{split}_pred{thr}_{name}', pred_path, make_new=False)
        ent = tif.imread(f'{ent_path}/{ent_name}.tif').squeeze()
        loop_score_gt, loop_score_pred = Graph.matching_by_nodes_iou(t, y, mode='cyc')[1:3]
        if y.n_cyc>0:
            tp_pos = y.cyc_cntr3[loop_score_pred == 1].astype('int').reshape(-1,3)
            fp_pos = y.cyc_cntr3[loop_score_pred != 1].astype('int').reshape(-1,3)
            for fp in fp_pos:
                ent_fp.append(ent[tuple(fp)])
            for tp in tp_pos:
                ent_tp.append(ent[tuple(tp)])
        if t.n_cyc>0:
##            tp_pos = y.cyc_cntr3[loop_score_pred == 1].astype('int').reshape(-1,3)
            fn_pos = t.cyc_cntr3[loop_score_gt != 1].astype('int').reshape(-1,3)
            for fn in fn_pos:
                ent_fn.append(ent[tuple(fn)])
##            for tp in tp_pos:
##                ent_tp.append(ent[tuple(tp)])

    data = [ent_tp, ent_fp, ent_fn]
    plt.figure()
    bp = plt.boxplot(data, showfliers=False, showmeans=True, meanline=True)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.xticks([1,2,3],['TP', 'FP', 'FN'])
    plt.ylabel('Entropies of predicted loop centers in validation set')
    plt.title(f'{ent_path.split("/")[3].capitalize()} entropy {title_2}\n predictions thresholded at {thr}')
    plt.savefig(f"{ent_path}/../loop {title_2}, pred{thr}.png")




def time_filtering(ys, t, threshold=3):
    # t = timepoint, indexed from 0 to N
    if t == 0:
        cmp_keys_yt_keep, cyc_keys_yt_keep = comparative_filter(ys, t, t+1, threshold)
    elif t == len(ys)-1:
        cmp_keys_yt_keep, cyc_keys_yt_keep = comparative_filter(ys, t, t-1, threshold)
    else:
        cmp_keys_yt_keep_prev, cyc_keys_yt_keep_prev = comparative_filter(ys, t, t-1, threshold)
        cmp_keys_yt_keep_next, cyc_keys_yt_keep_next = comparative_filter(ys, t, t+1, threshold)
        cmp_keys_yt_keep = list(set(cmp_keys_yt_keep_prev) | set(cmp_keys_yt_keep_next)) 
        cyc_keys_yt_keep = list(set(cyc_keys_yt_keep_prev) | set(cyc_keys_yt_keep_next))
        
    return cmp_keys_yt_keep, cyc_keys_yt_keep


def comparative_filter(ys, t, tother, threshold):

        yt = ys[t]
        ytother = ys[tother]

        cmp_keys_yt = yt.cmp_pos.keys()
        cyc_keys_yt = yt.cyc_pos.keys()

        cmp_keys_ytother = ytother.cmp_pos.keys()
        cyc_keys_ytother = ytother.cyc_pos.keys()

        cmp_keys_yt_keep = []
        cyc_keys_yt_keep = []

        for i in cmp_keys_yt:
            for j in cmp_keys_ytother:
                dist = pointcloud_dist(yt.cmp_pos[i], ytother.cmp_pos[j])
                if dist < threshold:
                    cmp_keys_yt_keep.append(i)
            for j in cyc_keys_ytother:
                dist = pointcloud_dist(yt.cmp_pos[i], ytother.cyc_pos[j])
                if dist < threshold:
                    cmp_keys_yt_keep.append(i)

        for i in cyc_keys_yt:
            for j in cmp_keys_ytother:
                dist = pointcloud_dist(yt.cyc_pos[i], ytother.cmp_pos[j])
                if dist < threshold:
                    cyc_keys_yt_keep.append(i)
            for j in cyc_keys_ytother:
                dist = pointcloud_dist(yt.cyc_pos[i], ytother.cyc_pos[j])
                if dist < threshold:
                    cyc_keys_yt_keep.append(i)

        return cmp_keys_yt_keep, cyc_keys_yt_keep

def pointcloud_dist(pointcloud1, pointcloud2):
    # Pointclouds in N x 3 format
    from scipy.spatial.distance import cdist
    D = cdist(pointcloud1, pointcloud2, metric='euclidean')
    dist = min(D.flatten())
    return dist

def draw_time_patches(target_path):
    import re
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    threshold = 3
    names = [name.replace('skel_label_val_', '').replace('.graph', '') for name in os.listdir(target_path) if name.endswith('.graph')]
    for name in names:
        print(name)
        ys = []
        res = temp.match(name.split('_')[1]).groups()
        
        for i in [-1,0,1]:
            t = int(res[1])+i
            n = name.split('_')
            n[1] = f'tp{t}'
            newname = '_'.join(n)
            graphname = f"pred0.7-{newname}"
            y = Graph(graphname, 'tp')
            ys.append(y)

        i = 1
        y_orig = ys[i]
        cmp_keys_yt_keep, cyc_keys_yt_keep = time_filtering(ys, i, threshold)
        
        cmp_keys_yt_keep_nodes = []
        for cmp in cmp_keys_yt_keep:
            cmp_keys_yt_keep_nodes = cmp_keys_yt_keep_nodes + y_orig.cmp[cmp]
            
        cyc_keys_yt_keep_nodes = []
        for cyc in cyc_keys_yt_keep:
            cyc_keys_yt_keep_nodes = cyc_keys_yt_keep_nodes + y_orig.cyc[cyc]
            
        keep_nodes = list(set(cyc_keys_yt_keep_nodes) | set(cmp_keys_yt_keep_nodes))
        seg = Graph(f'label_val_{name}', target_path)
        null = nx.Graph()
        null.add_nodes_from([0,1,2,3])
        pos_null={0:[0,0], 1:[0,255], 2:[255,0], 3:[255,255]}
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.title(f'{seg.name}_#L{seg.n_cyc}_#C{seg.n_cmp}')
        nx.draw(null, pos=pos_null, node_color='w', node_size=1e-10)
        nx.draw(seg.G, pos=seg.pos2, node_size=50, node_color='r', with_labels=False)
        nx.draw(seg.G, pos=seg.pos2, node_size=50, node_color='g', with_labels=False, nodelist=seg.cyclist)
        plt.subplot(122)
        plt.title(f'{y_orig.name}_#L{y_orig.n_cyc}_#C{y_orig.n_cmp}')
        nx.draw(null, pos=pos_null, node_color='w', node_size=1e-10)
        nx.draw(y_orig.G, pos=y_orig.pos2, node_size=50, node_color='k', with_labels=False, font_size=8)
        nx.draw(y_orig.G, pos=y_orig.pos2, node_size=50, node_color='r', with_labels=False, font_size=8, nodelist=cmp_keys_yt_keep_nodes)
        nx.draw(y_orig.G, pos=y_orig.pos2, node_size=50, node_color='gray', with_labels=False, font_size=8, nodelist=y_orig.cyclist)
        nx.draw(y_orig.G, pos=y_orig.pos2, node_size=50, node_color='g', with_labels=False, font_size=8, nodelist=cyc_keys_yt_keep_nodes)
        plt.savefig(f'{target_path}/{name}.png')

def draw_time_image(path):
    threshold = 1
    names = [name.replace('.tif', '') for name in os.listdir(path) if name.endswith('.tif')]
    ys = [Graph(f'one-ten_40_val_pred0.7_LI_2018-11-20_emb7_pos4_tp{t+1}', path) for t in range(len(names))]

    for i in range(len(ys)):
        y_orig = ys[i]
        cmp_keys_yt_keep, cyc_keys_yt_keep = time_filtering(ys, i, threshold)
        cmp_keys_yt_keep_nodes = []
        for cmp in cmp_keys_yt_keep:
            cmp_keys_yt_keep_nodes = cmp_keys_yt_keep_nodes + y_orig.cmp[cmp]
            
        cyc_keys_yt_keep_nodes = []
        for cyc in cyc_keys_yt_keep:
            cyc_keys_yt_keep_nodes = cyc_keys_yt_keep_nodes + y_orig.cyc[cyc]
            
        keep_nodes = set(cyc_keys_yt_keep_nodes) | set(cmp_keys_yt_keep_nodes)
        rm_nodes = set(y_orig.G) - keep_nodes
        y_orig.G.remove_nodes_from(list(rm_nodes))
        null = nx.Graph()
        null.add_nodes_from([0,1,2,3])
        pos_null={0:[0,0], 1:[0,1023], 2:[1023,0], 3:[1023,1023]}
        plt.figure(figsize=(10,10))
        plt.title(f'{y_orig.name}_#L{len(set(cyc_keys_yt_keep))}_#C{len(set(cmp_keys_yt_keep))}', y=-0.01)
        nx.draw(null, pos=pos_null, node_color='w', node_size=1e-10)
        nx.draw(y_orig.G, pos=y_orig.pos2, node_size=50, node_color='r', with_labels=False, nodelist=cmp_keys_yt_keep_nodes)
        nx.draw(y_orig.G, pos=y_orig.pos2, node_size=50, node_color='g', with_labels=False, nodelist=cyc_keys_yt_keep_nodes)
        os.makedirs(f'{path}/filtered', exist_ok=True)
        plt.savefig(f'{path}/filtered/{y_orig.name}.png', dpi=100)

def loop_center_csv(path, name, max_frame):
    center_frame = np.zeros([1,4])
    for frame in range(max_frame):
        s = Graph(f'{name}_tp{frame+1}', path)
        frame_appended = np.append(s.cyc_cntr3, np.tile(frame+1, s.n_cyc).reshape(-1,1), axis=1)
        center_frame = np.append(center_frame, frame_appended, axis=0)
    df = pd.DataFrame(center_frame[1:], columns=['z', 'y', 'x', 'frame'])
    df.to_csv(f'{path}/{name}.csv')
    
# thr_selection('D:/dataset/val/patches/label', 'results/unetcldice/2d/val', 'unetcldice', [200], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# print(os.listdir('LI_2018-11-20_emb7_pos4_preds'))
#loop_center_csv('E:/Skel/movie_pred/LI_2019-02-05_emb5_pos4/skel', 'pred0.7_LI_2019-02-05_emb5_pos4', 289)
##draw_time_image('LI_2018-11-20_emb7_pos4_preds')
##draw_time_image('time_toy')

##draw_time_change('labels_val')
##loop_ent('target_val_patches', 'm3/val/patches/calibrated/val_preds_patches','m3/val/patches/calibrated/val_ents_patches/blurred_ent_100', 'one-ten', 40, 'val', 0.3)
##loop_ent('target_val_patches', 'm3/val/patches/calibrated/val_preds_patches','m3/val/patches/calibrated/val_ents_patches/blurred_ent_50', 'one-ten', 40, 'val', 0.3)
##loop_ent('target_val_patches', 'm3/val/patches/calibrated/val_preds_patches','m3/val/patches/calibrated/val_ents_patches/blurred_ent_25', 'one-ten', 40, 'val', 0.3)
##get_difficulty()
##ent_target_pred('target_val_patches','m3/val/patches/calibrated/val_preds_patches', split='val', prefix='one-ten', thr_list=[0.3], epoch_list=[40])
##plot_ent_by_category('target_val_patches','m3/val/patches/uncalibrated/val_preds_patches', 'm3/val/patches/calibrated/val_ents_patches/blurred_ent_25', split='val', prefix='one-ten', thr_list=[0.7], epoch_list=[40])
##plot_ent_by_category(target_path='target_val_patches',pred_path='m3/val/patches/calibrated/val_preds_patches', ent_path='m3/val/patches/calibrated/val_ents_patches/blurred_ent_50', split='val', prefix='one-ten', thr_list=[0.3], epoch_list=[40])
##plot_hist_ent('m3/ts/ts_lins_patches/calibrated')    
##plot_hist_ent('m3/val/patches/calibrated/val_ents_patches/ent')
##convolve_entropy('m3/val/val_ents_patches', [25,25])
##x = np.load(f'E:/Skel/m3/val/val_ents_patches/conv_[100, 100]/one-ten_40_val_ent_LI_2016-03-04_emb5_pos2_tp105_A1D3D4_A2.tif.npy')
##print(x[0,:100,100:])

# y = Graph('one-ten_40_val_pred0.7_LI_2019-09-19_emb1_pos3_tp79_D1D2D3D4_C3', 'm3/val/val_preds_patches')
# y.ent_cyc()
##prob2ent('m3/val/val_lins_patches/calibrated')
##threshold_ent('m3/val/patches/calibrated/val_ents_patches/blurred_ent_25', np.arange(0.1,1,0.1))
##mean_ent('m3/ts/ts_lins_patches/calibrated')        
##add_prefix('m3/ts/ts_lins', 'one-ten')         
##make_patches('target_ts_patches', 'm3/ts/ts_lins', 'one-ten', 'ts', 40, 0.8)        
##      
##weighted_sig('m3/ts/ts_lins_patches', a=0.6634365, b=-1.3939593)
##calibration('target_ts_patches', 'm3/ts/ts_lins_patches/calibrated', 'one-ten', 40, 'ts')
##calibration('target_ts_patches', 'm3/ts/ts_probs_patches', 'one-ten', 40, 'ts')
##calibration('target_ts_patches','m3/ts/ts_probs_patches','m3/ts/ts_lins_patches/calibrated', 'one-ten', 40, 'ts', 'm3/ts/calibration')

##calibration('target_val_patches', 'm3/val/calibrated', 'one-ten', 40, 'val')
##tr_loss('m2/tr_loss', np.arange(1,201))
##select_random('target_ts_patches')
##names = ['LI_2015-07-12_emb5_pos1_tp16_D3D4_B2',
##            'LI_2019-06-13_emb2_pos2_tp31_A1D1_D4',
##            'LI_2019-07-03_emb7_pos2_tp97_D3D4_B4',
##            'LI_2019-06-13_emb2_pos2_tp31_A1D1_C2']
##for name in names:
##    t = Graph(f'one-ten_40_ts_pred0.7_{name}', 'm3/ts/ts_preds_patches', create_cyc_obj=True)

##t = Graph('seg_LI_2019-08-30_emb2_pos1_tp162_D2D3D4_A2', 'target_val_patches')
##t.plot_one_graph(mode='cyc')
##plt.show()
##tr_loss('m1/tr_loss', np.arange(1,201))
##mean_ent('m3/asdf')

##m1 = np.load(f'm1/ts/ts_preds_patches/score.npy')
##m2 = np.load(f'm2/ts/ts_preds_patches/score.npy')
##m3 = np.load(f'm3/ts/ts_preds_patches/score.npy')
##print(f'unet:{np.mean(m1, (0,2))[2]}±{np.std(m1, (0,2))[2]}, ae:{np.mean(m2, (0,2))[2]}±{np.std(m2, (0,2))[2]}, joint:{np.mean(m3, (0,2))[2]}±{np.std(m3, (0,2))[2]}')


##
##put_folder('m1/val/val_preds_patches/unet-normalized', '2d-unet', 199,'val',  np.arange(0.1, 1, 0.1))

##put_folder('m2/val/val_preds_patches/ae-normalized', 'ae_r20', 200,'val',  np.arange(0.1, 1, 0.1))
##threshold_prob_map('E:/phd_project/results/ae/2d/seg/images/prob/val/new_val/probs', np.arange(0.1, 1, 0.1))
##threshold_prob_map('C:/Users/arnav/desktop/probs', np.arange(0.1, 1, 0.1))
##threshold_prob_map('m3/val/patches/calibrated/val_probs_patches', [0.3])

##remove_s('m2/val/val_preds')
##for i in range(10,200):
##    make_patches('target_val_patches', 'm1/val/val_preds', '2d-unet', 'val', i, 0.9)
##threshold_prob_map('m3/ts/ts_probs', [0.7])
##thr_selection('target_ts_patches', 'm1/ts/ts_preds_patches', '2d-unet', [199], 'ts', [0.8])
##thr_selection('target_ts_patches', 'm2/ts/ts_preds_patches', 'ae_r20', [200], 'ts', [0.6])
##thr_selection('new_target_val_patches', 'm3/val/patches/uncalibrated/new', '', [40], 'val', np.arange(0.1,1.04,0.1))
##thr_selection('target_ts_patches', 'm3/ts/ts_preds_patches', 'one-ten', [23], 'val', [0.7], load=True)
##put_folder('m3/val/val_preds_patches/joint-normalized', 'one-ten', 40,'val',  np.arange(0.1, 1, 0.1))
       
##make_skels_path('m1/ts/ts_preds_patches')
##make_skels_path('C:/Users/arnav/desktop/preds') 
##for thr in np.arange(.1, 1, .1):

##t = Graph('one-ten_40_val_pred0.7_LI_2019-08-30_emb2_pos1_tp162_D2D3D4_A2', 'm3/val/val_preds_patches', make_new=False, create_cyc_obj=True)


##plot_recall_precision('m3/haus 35, iou 0.7', 40)
##pixel_early_stopping('val_patches_new', 'm3/val_patch_preds', 'one-ten', 40)

####edit_seg('one-ten_1_LI_2016-03-04_emb5_pos2_tp105_A1D3D4_A2', '.')

##prob2ent('m3/ts/patches/ts_lins_patches/calibrated')
# ent_target_pred('m3/ts/patches/ts_lins_patches/calibrated/ent', 'target_ts_patches', 'm3/ts/patches/ts_preds_patches', [0.7], [40], 'ts', 'one-ten')
thr_selection_pixel('D:/dataset/val/patches/label', 'results/unetcldice/2d/val', 'unetcldice', 200, 'val', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

