import numpy as np
import scipy.io as sio
import pandas as pd

class MAT2CSV:

    def __init__(self, path, name):
        self.path = path
        self.name = name

    def save_as_csv(self, path):
        for name in self.movie_names:
            if len(self.get_loop_idx(name))>0:
                df = self.numpy2df(name)
                df.to_csv(f'{path}/{name}.csv')
    

    def get_loop_events(self, name):
        loop_idx = self.get_loop_idx(name)
        return self.get_all_events(name)[:, loop_idx]

    def make_3d_loop_events(self, name):
        loop_events = self.get_loop_events(name)
        loop_events_3d = np.zeros([loop_events.shape[1], loop_events[0,0].shape[0], loop_events[0,0].shape[1]])
        for i in range(loop_events.shape[1]):
            loop_events_3d[i] = loop_events[0,i]
        return np.array(loop_events_3d)

    def append_frame_and_loop_id(self, name):
        loop_coor = self.make_3d_loop_events(name)
        num_loops, num_frame = loop_coor.shape[0:2]
        loop_coor_reshaped = loop_coor.reshape(-1, 2)
        frame = np.tile(np.arange(num_frame), num_loops).reshape(-1,1)
        loop_id = np.repeat(np.arange(num_loops), num_frame).reshape(-1,1)
        append_frame = np.append(loop_coor_reshaped, frame, axis=1)
        append_loop_id = np.append(append_frame, loop_id, axis=1)
        return append_loop_id

    def remove_nan(self, name):
        loop_coor = self.append_frame_and_loop_id(name)
        nan_rows = np.isnan(loop_coor).any(axis=1)
        return loop_coor[~nan_rows, :]

    def numpy2df(self, name):
        np_array = self.remove_nan(name)
        pd_df = pd.DataFrame(np_array, columns=['x', 'y', 'frame', 'loop_id'])
        return pd_df
        

    def get_name_idx(self, name):
        return int(np.argwhere(np.array(self.movie_names)==name))

    def get_num_events(self, name):
        return self.get_all_events(name).shape[1]

    def get_all_events(self, name):
        return self.mat['event'][0,self.get_name_idx(name)]['coor']
    
    def get_loop_idx(self, name):
        return [i for i in range(self.get_num_events(name)) if self.is_loop_event(name, i)]

    def is_loop_event(self, name, event_idx):
        event = self.mat['event'][0,self.get_name_idx(name)]['eventType'][0,event_idx]
        loop_type = ['h', 'L', 'LF', 'LC', 'LFLC', 'LFLB', 'LB']
        return np.isin(event, loop_type).any()

    @property
    def mat(self):
        return sio.loadmat(f'{self.path}/{self.name}.mat')[f'{self.name}']

    def __len__(self):
        return self.mat.shape[1]

    @property
    def movie_names(self):
        return [self.mat['path'][0,i][0] for i in range(len(self))]



if __name__ == '__main__':
    MAT2CSV('.', 'EM').save_as_csv('.')
