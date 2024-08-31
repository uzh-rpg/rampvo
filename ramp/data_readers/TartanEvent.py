from utils.transformers import EventSequenceToVoxelGrid_Pytorch, EventToStack_Numpy
from data import EventSequence, Events, H5EventHandle
from .augmentation import EventRGBDAugmentor
from ramp.utils import normalize_image
from .RGBDDataset import RGBDDataset
from .utils_data_readers import (
    EventSequenceToVoxelGrid_Pytorch,
    set_random_sequence_to_zero,
    set_random_sample_to_zero,
)
import numpy as np
from functools import partial
from copy import deepcopy
from pathlib import Path
import os.path as osp
from tqdm import tqdm
import pandas as pd
import torch
import glob


class TartanEvent(RGBDDataset):
    DEPTH_SCALE = 5.0  # scale depths to balance rot & trans

    def __init__(self, config, path, step=0, crop_size=[480, 640], workers_n=0, **kwargs):
        super(TartanEvent, self).__init__(config=config, return_indices=True, **kwargs)

        self.workers_n = workers_n
        train_cfg = config["data_loader"]["train"]["args"]
        self.ev_seq_params = {
            "height": train_cfg["image_height"],
            "width": train_cfg["image_width"],
        }
        self.event_representation_type = config["event_representation"]
        self.num_events_selected = train_cfg["num_events_selected"]

        self.data_drop = train_cfg["data_drop"]
        self.default_prob = [0.4, 0.4, 0.2]
        if train_cfg.get("data_drop_prob"):
            self.data_drop_prob = train_cfg["data_drop_prob"]

        self.stepsto_finetune = 1000
        if train_cfg.get("steps_until_finetune"):
            self.stepsto_finetune = train_cfg["steps_until_finetune"]

        self.norm_img_to = None
        if train_cfg.get("norm_img_to"):
            self.norm_img_to = train_cfg["norm_img_to"]

        self.step_batch_map = None
        if train_cfg.get("step_batch_map"):
            self.step_batch_map = train_cfg["step_batch_map"]
        
        self.n_events_in_between = 0
        if train_cfg.get("n_events_in_between"):
            self.n_events_in_between = train_cfg["n_events_in_between"]

        self.events_importing_mode = None
        if train_cfg.get("events_importing_mode"):
            self.events_importing_mode = train_cfg["events_importing_mode"]

        self.num_event_bins = train_cfg["num_event_bins"]
        self.aug = train_cfg["augment_data"]
        self.type = train_cfg["type"]
        self.root = path
        self.iter = step

        if self.event_representation_type == "voxels":
            self.event_representation = EventSequenceToVoxelGrid_Pytorch(
                num_bins=self.num_event_bins, normalize=True, gpu=True
            )
            self.to_event_sequence = partial(
                EventSequence,
                params=self.ev_seq_params,
                timestamp_multiplier=None,
                convert_to_relative=True,
            )
        elif self.event_representation_type == "stack":
            self.event_representation = EventToStack_Numpy(
                num_bins=self.num_event_bins
            )
        else:
            raise NotImplementedError

        if self.aug:
            self.augmentor = EventRGBDAugmentor(crop_size=crop_size)

        self.build_events_indices()

    @staticmethod
    def check_indices(indices_file):
        if not indices_file.exists():
            raise ValueError(
                f"Indices file does not exists in {indices_file},  rename it to indices.txt  or check if you computed it correctly"
            )

    def build_events_indices(self):
        # Preload events indices
        # ...and add validation indices
        self.i0, self.i1 = {}, {}
        all_indices = deepcopy(self.dataset_index)

        for scene_id_path in self.validation_index:
            all_indices.append(scene_id_path)

        for scene_data in all_indices:
            scene_id_path = scene_data if isinstance(scene_data, str) else scene_data[0]

            if (
                self.i0.get(scene_id_path) is None
                and self.i1.get(scene_id_path) is None
            ):
                indices_file = Path(scene_id_path) / Path("indices.txt")
                self.check_indices(indices_file)
                i0, i1 = np.loadtxt(indices_file, delimiter=",").astype(int)
                self.i0[scene_id_path] = i0
                self.i1[scene_id_path] = i1

    def _build_dataset(self):
        print("Building TartanEvent dataset")
        scene_info = {}
        scenes = glob.glob(osp.join(self.root, "*/*/*/*"))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, "image_left/*.png")))
            depths = sorted(glob.glob(osp.join(scene, "depth_left/*.npy")))
            events = sorted(glob.glob(osp.join(scene, "events/*.h5")))

            if len(images) != len(depths):
                continue

            poses = np.loadtxt(osp.join(scene, "pose_left.txt"), delimiter=" ")
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:, :3] /= self.DEPTH_SCALE
            intrinsics = [TartanEvent.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = "/".join(scene.split("/"))
            scene_info[scene] = {
                "events": events,
                "images": images,
                "depths": depths,
                "poses": poses,
                "intrinsics": intrinsics,
                "graph": graph,
            }
        return scene_info

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanEvent.DEPTH_SCALE
        depth[depth == np.nan] = 1.0
        depth[depth == np.inf] = 1.0
        return depth

    @staticmethod
    def get_event_by_index(idx, event, i0, i1):
        if isinstance(event, np.ndarray):
            return event.get_between_idx(i0[idx], i1[idx]).to_array()
        elif isinstance(event, H5EventHandle):
            return event.get_between_idx(i0[idx], i1[idx])
        else:
            print("unrecognized event type")
            raise ValueError

    @staticmethod
    def change_dataframe_column_order(df):
        cols = np.array(df.columns.values)
        idx = [2, 0, 1, -1]
        return df[cols[idx]]

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def is_test_scene(scene, test_scenes):
        return any(x in scene for x in test_scenes)

    @staticmethod
    def normalize_depth_and_poses(poses, disps):
        s = 0.7 * torch.quantile(disps, 0.98)
        disps = disps / s
        poses[..., :3] *= s
        return poses, disps

    def event_representation_from_event(self, event_blob):
        if self.event_representation_type == "voxels" and isinstance(
            event_blob, np.array
        ):
            evframe = pd.DataFrame(event_blob, columns=["x", "y", "ts", "p"])
            event_frame = self.change_dataframe_column_order(evframe)
            event_sequence = self.to_event_sequence(dataframe=event_frame)
            return self.event_representation(event_sequence)
        elif self.event_representation_type == "stack" and isinstance(
            event_blob, Events
        ):
            return torch.tensor(self.event_representation(event_blob))
        else:
            print("unrecognized event representation or wrong input data type")
            raise NotImplementedError

    def events_from_path(self, inds, path, scene_info):
        self.event = H5EventHandle.from_path(Path(scene_info[path]["events"]))
        i0, i1 = self.i0[path], self.i1[path]

        all_events = []
        for i in inds:
            event_blob = self.get_event_by_index(idx=i, event=self.event, i0=i0, i1=i1)
            event_tensor = self.event_representation_from_event(event_blob)
            all_events.append(event_tensor)
        return torch.stack(all_events, dim=0)

    def events_from_indices(self, event, i_start, i_stop):
        event_blob = event.get_between_idx(i_start, i_stop)
        event_tensor = self.event_representation_from_event(event_blob)
        return event_tensor

    @staticmethod  
    def transform_data(images, events, poses, depths, intrinsics, zeroed_mask):

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)
        events = torch.stack(events, dim=0)
        zeroed_mask = torch.tensor(zeroed_mask)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        return images, events, poses, disps, intrinsics, zeroed_mask

    def get_data_from_inds(self, data_index):
        """ return training video """
        # TODO LOG THE PARSED INDICES TO SEE IF THE ENTIRE DATASET IS SPANNED
        # TODO TRY ADDING MORE EVENTS
        # TODO REFACTOR THE CODE 
        inds, scene_id = self.get_indices_to_load(data_index)

        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']
        event_file = self.scene_info[scene_id]['events']
        self.event = H5EventHandle.from_path(Path(event_file))
        i1 = self.i1[scene_id]

        images, depths, poses, intrinsics, events = [], [], [], [], []
        supervision_mask = []
        n_loaded = 0
        if self.events_importing_mode == "all_events_all_images":
            # Load all events in between images and all images and see how it goes
            for index in range(min(inds), max(inds)+1):

                events_stream_size = (i1[index]-i1[index-1])
                segments_size = events_stream_size // (self.n_events_in_between+1)

                for i in range(self.n_events_in_between+1):
                    first_ind = i1[index-1] + segments_size*i
                    last_ind = first_ind + segments_size
                    try:
                        event_tensor = self.events_from_indices(self.event, first_ind, last_ind)
                    except:
                        print(f"error cannot import events from {scene_id} index {index}")
                        event_tensor = torch.zeros((self.num_event_bins, self.ev_seq_params["height"], self.ev_seq_params["width"]))
                    events.append(event_tensor)

                    supervise = True if i == self.n_events_in_between else False
                    supervision_mask.append(supervise)

                images.append(self.__class__.image_read(images_list[index]))
                depths.append(self.__class__.depth_read(depths_list[index]))
                poses.append(poses_list[index])
                intrinsics.append(intrinsics_list[index])
            
                n_loaded += 1
                if n_loaded == self.n_frames:
                    break
        else:
            # add events in between images from precedent frame to current frame divided in chunks
            for index in inds:
                events_stream_size = (i1[index]-i1[index-1])

                if index == inds[0]:
                    event_stream_chunks_n = 1
                else:
                    event_stream_chunks_n = events_stream_size//self.num_events_selected
                    
                first_ind = i1[index-1] + events_stream_size % self.num_events_selected
                for stream_ind in range(event_stream_chunks_n-1):
                    if stream_ind >= self.n_events_in_between:
                        break
                    last_ind = first_ind + self.num_events_selected
                    event_tensor = self.events_from_indices(self.event, first_ind, last_ind)
                    events.append(event_tensor)
                    supervision_mask.append(False)
                    first_ind = last_ind
                # Load the event corresponding to the current frame all of the same size
                first_ind = i1[index]-self.num_events_selected
                last_ind = i1[index]
                event_tensor = self.events_from_indices(self.event, first_ind, last_ind)
                events.append(event_tensor)
                supervision_mask.append(True)
    
                images.append(self.__class__.image_read(images_list[index]))
                depths.append(self.__class__.depth_read(depths_list[index]))
                poses.append(poses_list[index])
                intrinsics.append(intrinsics_list[index])
                n_loaded += 1
                if n_loaded == self.n_frames:
                    break
        
        return self.transform_data(images, events, poses, depths, intrinsics, supervision_mask)

    def __getitem__(self, idx):
        self.iter += self.workers_n
        if idx == 0: 
        # skip first element, since events are generated by subsequent images contrast difference.
            return 0

        if self.step_batch_map is not None and str(self.iter) in self.step_batch_map.keys():
            self.n_frames = self.step_batch_map[str(self.iter)]

        images, events, poses, disps, intrinsics, supervision_mask = self.get_data_from_inds(idx)

        if self.aug:
            events, images, poses, disps, intrinsics = self.augmentor(events, images, poses, disps, intrinsics)

        poses, disps = self.normalize_depth_and_poses(poses=poses, disps=disps)
        images = normalize_image(images=images, norm_img_to=self.norm_img_to)

        if self.data_drop == "sample_drop":
            events, images = set_random_sample_to_zero(events=events, images=images)
            
        # TODO Do not restart to train a model with sequence drop
        elif self.data_drop == "sequence_drop" and self.iter >= self.stepsto_finetune:
            events, images = set_random_sequence_to_zero(
                perc_to_drop_img=self.data_drop_prob[0],
                perc_to_drop_evs=self.data_drop_prob[1],
                perc_to_drop_none=self.data_drop_prob[2],
                events=events,
                images=images,
            )
        
        return (
            events.float(),
            images.float(),
            poses.float(),
            disps.float(),
            intrinsics.float(),
            supervision_mask,
        )
