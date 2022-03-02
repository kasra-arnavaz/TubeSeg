from pygel3d import graph, hmesh
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import numpy as np
import os
import tifffile as tif
from argparse import ArgumentParser

class Skeleton:

    def __init__(self, seg_path: str, name: str, make_new: bool=False, make_obj: bool=False, write_path: str=None) -> None:
        ''' Makes skeletons out of binary predictions
        seg_path: path to binary segmentation images; predicted or label
        name: name of the binary segmentation tif file
        make_new: if True, will overwrite existing skeleton
        make_obj: if True, will make .obj file which can visualize the skeleton in MeshLab app. Saved in ./{write_path}/obj
        write_path: the main path to save results in. Default './{seg_path}/..'. Skeletons are saved in ./{write_path}/skel
        '''
        if write_path is None:
            self.write_path = f'{seg_path}/..'
        else:
            self.write_path = write_path
        self.seg_path = seg_path
        self.name = name
        self.make_new = make_new
        self.make_obj = make_obj
        self.run()


    def run(self):
        ''' The main method that runs every step of the skeletonization algorithm
        '''
        if self.skeleton_deos_not_exist() or self.make_new or self.skeleton_aborted():
            print(f'Making new skeleton for {self.name}')
            self.write_graph()
            self.skeletonize()
            self.scale_skeleton()
            if self.make_obj: self.write_obj()
            self.remove_graph()

    def skeleton_deos_not_exist(self):
        return f'{self.name}.skel' not in os.listdir(self.write_path)

    def skeleton_aborted(self):
        return os.path.getsize(f'{self.write_path}/skel/{self.name}.skel') < 1000
        
    def load_tif(self, crop=3):
        ''' crops the segmentations slightly to get rid of foreground pixels in the label images due to imperfect patching
        '''
        return tif.imread(f'{self.seg_path}/{self.name}.tif')[:,crop:-crop, crop:-crop]

    def get_positions_and_edges(self):
        ''' Uses kNN to convert segmentations into nodes and edges
        '''
        tif = self.load_tif()
        positions = np.argwhere(tif).astype(np.float32)
        neigh = NearestNeighbors(n_neighbors=4, radius=1.9, metric='l2')
        neigh.fit(positions)
        radius_neighbors_graph = neigh.radius_neighbors_graph(positions).astype(np.bool)
        i, j, _ = sparse.find(sparse.triu(radius_neighbors_graph, k=1))
        edges = np.asarray([i, j]).transpose()
        return positions, edges

    def write_graph(self):
        ''' Converts the segmentation into a graph.
        '''
        os.makedirs(f'{self.write_path}/skel', exist_ok=True)
        positions, edges = self.get_positions_and_edges()
        with open(f'{self.write_path}/skel/{self.name}.graph','w') as f:
                for p in positions:
                    print("n", p[0], p[1], p[2], file=f)
                for e in edges:
                    print("c", e[0], e[1], file=f)

    def skeletonize(self):
        ''' Makes the skeleton out of the produced graph.
        '''
        g = graph.load(f'{self.write_path}/skel/{self.name}.graph')
        graph.edge_contract(g, 3)
        s = graph.LS_skeleton(g)
        graph.prune(s)
        graph.save(f'{self.write_path}/skel/{self.name}.skel', s)
    
    def scale_skeleton(self):
        ''' Sets the correct scales along the z axis.
        '''
        with open(f'{self.write_path}/skel/{self.name}.skel') as f:
            lines = f.read().splitlines()
        with open(f'{self.write_path}/skel/{self.name}.skel', 'w') as f:
            for line in lines:
                if line.startswith('n') and 'nan' not in line:
                    print(line.split(' ')[0], float(line.split(' ')[1])*(2.89),float(line.split(' ')[2]), float(line.split(' ')[3]), file=f)
                else: print(line, file=f)
    
    def write_obj(self):
        os.makedir(f'{self.write_path}/obj')
        s = graph.load(f'{self.write_path}/skel/{self.name}.skel')
        m = graph.to_mesh_cyl(s, fudge=0.5)
        hmesh.obj_save(f'{self.write_path}/obj/{self.name}.obj', m)

    def remove_graph(self):
        ''' Removes the graph that is converted from the segmentation to save space.
        '''
        os.remove(f'{self.write_path}/skel/{self.name}.graph')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seg_path', type=str)
    parser.add_argument('--make_new', action='store_true', default=False)
    parser.add_argument('--make_obj', action='store_true', default=False)
    args = parser.parse_args()
    tif_names = [file.replace('.tif', '') for file in os.listdir(args.seg_path) if file.endswith('.tif')]
    for tif_name in tif_names:
        Skeleton(args.seg_path, tif_name, args.make_new, args.make_obj)
