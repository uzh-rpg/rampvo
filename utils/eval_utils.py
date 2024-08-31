import os
import torch
import numpy as np
from copy import deepcopy
from evo.core import sync
import evo.main_ape as main_ape
from matplotlib import pyplot as plt
from evo.tools import file_interface
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(quaternion):
    # Create a Rotation object from the quaternion
    # quaternion (qx, qy, qz, qw)
    r = R.from_quat(quaternion)
    
    # Convert the rotation to Euler angles (in radians)
    euler = r.as_euler('xyz')  # Adjust 'xyz' according to your convention (e.g., 'xyz', 'zyx', etc.)
    
    return euler

def ate(traj_ref, traj_est, timestamps):
    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:][:, (1,2,3,0)],
        timestamps=timestamps,
    )
    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:, :3],
        orientations_quat_wxyz=traj_ref[:, 3:],
        timestamps=timestamps,
    )
    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    return result.stats["rmse"]


def rot_err(traj_ref, traj_est, timestamps):
    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:],
        timestamps=timestamps,
    )
    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:, :3],
        orientations_quat_wxyz=traj_ref[:, 3:],
        timestamps=timestamps,
    )
    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.rotation_part,
        align=True,
        correct_scale=True,
    )
    return result.stats["rmse"]


def plot_events(event, image, i0, i1, i):
    # plotting
    plt.clf()
    if not os.path.exists("/home/pellerito/DPVO/image_ev_alignment"):
        os.makedirs("/home/pellerito/DPVO/image_ev_alignment")
    event_struct = event.get_between_idx(i0, i1)
    rendered = event_struct.render(image.permute(1, 2, 0).numpy())
    plt.imshow(rendered)
    plt.savefig("/home/pellerito/DPVO/image_ev_alignment/rendered_{:05d}.png".format(i))


def plot_events_voxels(events, image, i):
    plt.clf()
    if not os.path.exists("/home/pellerito/DPVO/image_ev_alignment"):
        os.makedirs("/home/pellerito/DPVO/image_ev_alignment")
    events_ = deepcopy(events)

    # events[events<0]=0 
    # events = events.sum(0)
    events[events<0]=1
    events = events.sum(0)

    events_[events_>0]=0
    events_[events_<0]=1
    events_ = events_.sum(0)
    # events_rgb = torch.stack((events*255, torch.zeros_like(events), events_*255), dim=0).permute(1,2,0)
    events_rgb = torch.stack((events*255, events, events), dim=0).permute(1,2,0)
    image = image.permute(1, 2, 0)
    plt.imshow(events_rgb)
    plt.imshow(image, alpha=0.5)
    plt.savefig("/home/pellerito/DPVO/image_ev_alignment/rendered_{:05d}.png".format(i))


def select_scene_cut(data_list, traj_ref, scene_path):
    if "indoor_flying4" in scene_path:
        data_list = data_list[160:]
        traj_ref = traj_ref[160:]
    elif "indoor_flying3" in scene_path:
        data_list = data_list[189:]
        traj_ref = traj_ref[189:]
    elif "indoor_flying2" in scene_path:
        data_list = data_list[250:]
        traj_ref = traj_ref[250:]
    elif "indoor_flying1" in scene_path:
        data_list = data_list[105:]
        traj_ref = traj_ref[105:]
    else:
        data_list = data_list
        traj_ref = traj_ref
    return data_list, traj_ref


def read_eds_format_poses(traj_path):
    mat = np.array(np.loadtxt(traj_path)).astype(float)
    stamps = mat[:, 0]  # n x 1
    xyz = mat[:, 1:4]  # n x 3
    quat = mat[:, 4:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    traj_ref = PoseTrajectory3D(xyz, quat, stamps)
    return traj_ref

def read_stereodavis_format_poses(traj_path, timestamps_path):
    mat = np.array(np.loadtxt(traj_path)).astype(float)
    timestamps = np.loadtxt(timestamps_path)/1e6
    xyz = mat[:, 0:3]  # n x 3
    quat = mat[:, 3:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    traj_ref = PoseTrajectory3D(xyz, quat, timestamps)
    return traj_ref

def read_tartan_format_poses(traj_path, timestamps_path):
    NED_TO_XYZ_PERM = [1, 2, 0, 4, 5, 3, 6] 
    traj = np.loadtxt(traj_path, delimiter=" ")[1:, NED_TO_XYZ_PERM]

    stamps = np.loadtxt(timestamps_path) # n x 1
    xyz = traj[:, 0:3]  # n x 3
    quat = traj[:, 3:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    traj_ref = PoseTrajectory3D(xyz, quat, stamps)
    return traj_ref

def read_moonlanding_format_poses(traj_path, timestamps_path):
    NED_TO_XYZ_PERM = [1, 2, 0, 4, 5, 3, 6] 
    traj = np.loadtxt(traj_path, delimiter=" ")[1:, NED_TO_XYZ_PERM]
    stamps = np.loadtxt(timestamps_path) # n x 1
    xyz = traj[:, 0:3]  # n x 3
    quat = traj[:, 3:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    traj_ref = PoseTrajectory3D(xyz, quat, stamps)
    return traj_ref