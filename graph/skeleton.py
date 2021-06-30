from PyGEL3D import gel
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import numpy as np
import os
import tifffile as tif


class Skeleton:

    def __init__(self, path, name, make_new=False):
        self.path = path
        self.name = name
        self.make_new = make_new
        self.run()


    def run(self):
        if self.skeleton_deos_not_exist() or self.make_new:
            print(f'Making new skeleton for {self.name}')
            self.write_graph()
            self.skeletonize()
            self.scale_skeleton()
            self.write_obj()
            self.remove_graph()

    def skeleton_deos_not_exist(self):
        return f'{self.name}.skel' not in os.listdir(self.path)
        
    def load_tif(self, crop=3):
        return tif.imread(f'{self.path}/{self.name}.tif')[:,crop:-crop, crop:-crop]

    def get_positions_and_edges(self):
        tif = self.load_tif()
        positions = np.argwhere(tif).astype(np.float32)
        neigh = NearestNeighbors(n_neighbors=4, radius=1.9, metric='l2')
        neigh.fit(positions)
        radius_neighbors_graph = neigh.radius_neighbors_graph(positions).astype(np.bool)
        i, j, _ = sparse.find(sparse.triu(radius_neighbors_graph, k=1))
        edges = np.asarray([i, j]).transpose()
        return positions, edges

    def write_graph(self):
        positions, edges = self.get_positions_and_edges()
        with open(f'{self.path}/{self.name}.graph','w') as f:
                for p in positions:
                    print("n", p[0], p[1], p[2], file=f)
                for e in edges:
                    print("c", e[0], e[1], file=f)

    def skeletonize(self):
        g = gel.graph_load(f'{self.path}/{self.name}.graph')
        gel.graph_edge_contract(g, 3)
        s = gel.graph_LS_skeleton(g)
        gel.graph_prune(s)
        gel.graph_save(f'{self.path}/{self.name}.skel', s)
    
    def scale_skeleton(self):
        with open(f'{self.path}/{self.name}.skel') as f:
            lines = f.read().splitlines()
        with open(f'{self.path}/{self.name}.skel', 'w') as f:
            for line in lines:
                if line.startswith('n') and 'nan' not in line:
                    print(line.split(' ')[0], float(line.split(' ')[1])*(2.89),float(line.split(' ')[2]), float(line.split(' ')[3]), file=f)
                else: print(line, file=f)
    
    def write_obj(self):
        s = gel.graph_load(f'{self.path}/{self.name}.skel')
        m = gel.graph_to_mesh_cyl(s, fudge=0.5)
        gel.obj_save(f'{self.path}/{self.name}.obj', m)

    def remove_graph(self):
        os.remove(f'{self.path}/{self.name}.graph')

if __name__ == '__main__':
    name = 'LI_2019-11-08_embX_pos1'
    path = f'movie/val/{name}/pred'
    tifs = [tif.replace('.tif', '') for tif in os.listdir(path) if tif.endswith('.tif')]
    for t in tifs:
        Skeleton(path, t)
