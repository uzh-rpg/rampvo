import numpy as np
import torch
import torch.utils.data as data

import os
import cv2
import pickle
import os.path as osp
from .rgbd_utils import *

class RGBDDataset(data.Dataset):
    def __init__(self, config, return_indices=False, fmin=10.0, fmax=75.0):
        """ Base class for RGBD dataset """

        self.return_indices=return_indices
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples

        train_cfg = config["data_loader"]["train"]["args"]
        self.sample = train_cfg["load_sampled_frames"]
        self.n_frames = train_cfg["n_frames"]
        self.test_scenes=config["data_loader"]["test"]["test_split"]
        path_pickle_dataset=config["path_pickle_dataset"]

        # building dataset is expensive, cache such that it only needs to be performed once
        cur_path = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))

        self.scene_info = self.__build_dataset(path=path_pickle_dataset)
        self._build_dataset_index()

    def __build_dataset(self, path):
        return pickle.load(open(path, 'rb'))
    
    def _build_dataset(self, path):
        return self.__build_dataset(path)
                
    def _build_dataset_index(self):
        self.dataset_index = []
        self.validation_index = []
        for scene in self.scene_info:
            if not self.__class__.is_test_scene(scene=scene, test_scenes=self.test_scenes):
                graph = self.scene_info[scene]['graph']
                for i in graph:
                    if i < len(graph) - 65:
                        self.dataset_index.append((scene, i))
            else:
                self.validation_index.append(scene)
        
        # if external validation # TODO merge with the previous code
        if not self.validation_index:
            for scene in self.test_scenes:
                self.validation_index.append(scene)

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        raise NotImplementedError

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph
    
    def get_indices_to_load(self, index):
        index = index % len(self.dataset_index)
        scene_id, frame_ix = self.dataset_index[index]
        self.scene_id = scene_id

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']

        d = np.random.uniform(self.fmin, self.fmax)
        s = 1

        i1 = self.i1[scene_id]
        how_many_events = i1[1:]-i1[:-1]

        inds = [ frame_ix ]

        while len(inds) < self.n_frames:
            # get other frames within flow threshold
            if self.sample:
                k = (frame_graph[frame_ix][1] > self.fmin) & (frame_graph[frame_ix][1] < self.fmax)
                frames = frame_graph[frame_ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > frame_ix]):
                    frame_ix = np.random.choice(frames[frames > frame_ix])
                elif frame_ix + 1 < len(images_list):
                    frame_ix = frame_ix + 1
                elif np.count_nonzero(frames):
                    frame_ix = np.random.choice(frames)
                
                if frame_ix <=0:
                    continue
                if how_many_events[frame_ix-1] < 0:
                    continue
            else:
                # TODO understand this code
                i = frame_graph[frame_ix][0].copy()
                g = frame_graph[frame_ix][1].copy()

                g[g > d] = -1
                if s > 0:
                    g[i <= frame_ix] = -1
                else:
                    g[i >= frame_ix] = -1

                if len(g) > 0 and np.max(g) > 0:
                    frame_ix = i[np.argmax(g)]
                else:
                    if frame_ix + s >= len(images_list) or frame_ix + s < 0:
                        s *= -1

                    frame_ix = frame_ix + s
            
            inds += [ frame_ix ]
        
        return inds, scene_id


    def __basegetitem__(self, index):
        """ return training video """
        inds, scene_id = self.get_indices_to_load(index)

        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        images, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.return_indices:
            return images, poses, disps, intrinsics, inds
        return images, poses, disps, intrinsics 

    def __getitem__(self, index):
        return self.__basegetitem__(index)
    
    def __len__(self):
        return len(self.dataset_index)-1 # TODO add flag skip 0 frame
    
    def __imul__(self, x):
        self.dataset_index *= x
        return self
