import os

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

import sys
import glob
import yaml
import json
import torch
import argparse
import torchvision
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from evo.core import sync
from functools import partial
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from utils.seed_everything import seed_everything
from ramp.data_readers.TartanEvent import TartanEvent
from utils.rotation_error_with_euler import rot_error_with_alignment_from_pose3d
from utils.eval_utils import (
    read_eds_format_poses,
    read_stereodavis_format_poses,
    read_tartan_format_poses,
    read_moonlanding_format_poses
)
from data import H5EventHandle
from ramp.utils import (
    input_resize,
    normalize_image,
    save_output_for_COLMAP
)
from ramp.config import cfg as VO_cfg
from ramp.Ramp_vo import Ramp_vo


seed_everything(seed=1234)
sys.setrecursionlimit(100000)


def set_global_params(K_path=None, standard_pose_format=False, resize_to=None):
    global fx, fy, cx, cy

    if K_path is None or not os.path.exists(K_path):
        fx, fy, cx, cy = [320, 320, 320, 240]
        print("Using default intrinsics", [fx, fy, cx, cy])
        return (fx, fy, cx, cy)
    else:
        # Load the YAML file
        with open(K_path, "r") as file:
            data = yaml.safe_load(file)

        # Extract the intrinsics
        intrinsics = data["cam0"]["intrinsics"]

        # Extract the individual components
        fx, fy, cx, cy = intrinsics

    if resize_to is not None:
        resolution = data["cam0"]["resolution"]
        slack = np.array(resize_to) - np.array(resolution)
        d_cx, d_cy = slack[0] / 2, slack[1] / 2
        cx = cx + d_cx
        cy = cy + d_cy

    print("Using intrinsics from {}".format(K_path), (fx, fy, cx, cy))
    return (fx, fy, cx, cy)


def save_results(
    traj_ref, traj_est, scene, j=0, eval_type="None"
):
    # save poses for finer evaluations
    save_dir = osp.join(
        os.getcwd(),
        "trajectory_evaluation",
        f"{eval_type}",
        "trial_" + str(j),
        scene,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_ref = (traj_ref.timestamps * 10 ** -9)[..., np.newaxis]
    time_est = (traj_est.timestamps * 10 ** -9)[..., np.newaxis]
    np.savetxt(
        osp.join(save_dir, "stamped_groundtruth.txt"),
        np.concatenate((time_ref, traj_ref.positions_xyz, traj_ref.orientations_quat_wxyz), axis=1),
    )
    np.savetxt(
        osp.join(save_dir, "stamped_traj_estimate.txt"),
        np.concatenate((time_est, traj_est.positions_xyz, traj_est.orientations_quat_wxyz), axis=1),
    )


def data_loader_all_events(
    config, full_scene, downsample_fact=1, norm_to=None, extension=".png"
):
    images_paths = osp.join(full_scene, "image_left", "*{}".format(extension))
    imfiles = sorted(glob.glob(images_paths))
    evfile = osp.join(full_scene, "events.h5")
    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    TartanEvent_loader = TartanEvent(config=config, path=full_scene)
    timestamps = np.loadtxt(osp.join(full_scene, "timestamps.txt"))

    # skip first element (no events for it)
    image_files = sorted(imfiles)[1 :: downsample_fact]
    corresponding_timestamps = timestamps[1 :: downsample_fact]

    # load events and compute how many are they
    event = H5EventHandle.from_path(Path(evfile))
    n_events = len(event.t)
    n_events_selected = TartanEvent_loader.num_events_selected
    n_events_voxels = n_events // n_events_selected
    corr_events_timestamps = event.t[0:n_events:n_events_selected][1::]

    time_vicinity = (
        np.subtract.outer(corr_events_timestamps, corresponding_timestamps) ** 2
    )
    corresponding_frame_indices = np.argmin(time_vicinity, axis=1)
    corresponding_events_indices = np.argmin(time_vicinity, axis=0)

    print("import images and events ...")
    data_list = []
    masks = []
    i1 = 0
    for i in tqdm(range(n_events_voxels)):
        i0 = i1
        i1 = i1 + n_events_selected
        event_voxel = TartanEvent_loader.events_from_indices(
            event=event, i_start=i0, i_stop=i1
        )

        frame_ind = corresponding_frame_indices[i]
        imfile = image_files[frame_ind]
        image = torchvision.io.read_image(imfile)
        image = normalize_image(images=image, norm_img_to=norm_to)

        # plot_events(event, image, i0, i1, i)
        # the index of the smallest error between the event voxel timestamp and the image timestamp is event index
        event_ind = corresponding_events_indices[frame_ind]
        if event_ind == i:
            mask = True
        else:
            mask = False
        masks.append(mask)
        data_list.append((image, event_voxel, intrinsics, torch.tensor([mask])))

    # frame_indices = list(set(corresponding_frame_indices))
    # Check this masking operation
    frame_indices = list(set(corresponding_frame_indices[masks]))
    return data_list, frame_indices


def _data_iterator(data_list):
    for image, events, intrinsics, mask in data_list:
        im = image[None, None, ...].cuda()
        ev = events[None, None, ...].float().cuda()
        intr = intrinsics.cuda()
        mask.cuda()
        yield im, ev, intr, mask


def resize_input(image, events):
    default_shape = torch.tensor([480, 640])
    data_shape = image.shape[-2:]
    if data_shape != default_shape:
        image, events = input_resize(
            image, events, desired_ht=data_shape[0] + 1, desired_wh=data_shape[1] + 1
        )

    image = (
        torch.stack((image, image, image), dim=3)[0, ...]
        if image.shape[-3] == 1
        else image
    )
    image.squeeze(0).squeeze(0)
    return image, events


@torch.no_grad()
def run_pose_pred(cfg_VO, network, eval_cfg, data_list, t_horizon_to_pred, t_to_pred, deg_approx=4):
    """Run the slam on the given data_list using pose prediction algorithm 
       for bootstrapping and return the trajectory and timestamps. 
       Pose prediction typically slows down VO frequency.
        
    Args:
        cfg_VO: config for the slam
        network: the network to use for the slam
        eval_cfg: config for the evaluation
        data_list: list of tuples (image, events, intrinsics)
        t_horizon_to_pred: the time horizon to predict the future
        t_to_pred: the time to start predicting the future
        deg_approx: the degree of the polynomial to use for the prediction
        
    Returns:
        traj_est: the estimated trajectory
        tstamps: the timestamps of the estimated trajectory
    """
    train_cfg = eval_cfg["data_loader"]["train"]["args"]
    slam = Ramp_vo(cfg=cfg_VO, network=network, train_cfg=train_cfg)
    for t, (image, events, intrinsics, mask) in enumerate(tqdm(_data_iterator(data_list))):

        image, events = resize_input(image, events)

        if t < t_to_pred or t_to_pred < 0:
            slam(t, input_tensor=(events, image, mask), intrinsics=intrinsics)
            last_keyframe_number = slam.n
        if t == t_to_pred and t_to_pred > 0:
            for _ in range(12):
                slam.update()
        if t >= t_to_pred and t_to_pred > 0:
            sec_to_pred_future = t - t_to_pred
            slam.predict_future_pose(
                    last_keyframe_number=last_keyframe_number,
                    sec_to_pred_future=sec_to_pred_future, 
                    abs_time=t,
                    deg=deg_approx,
                    )
        if t == t_to_pred + t_horizon_to_pred:
            break

    for _ in range(12):
        slam.update()

    return slam.terminate()


@torch.no_grad()
def run(cfg_VO, network, eval_cfg, data_list):
    """Run the slam on the given data_list and return the trajectory and timestamps

    Args:
        cfg_VO: config for the slam
        network: the network to use for the slam
        eval_cfg: config for the evaluation
        data_list: list of tuples (image, events, intrinsics)

    Returns:
        traj_est: the estimated trajectory
        tstamps: the timestamps of the estimated trajectory
    """
    train_cfg = eval_cfg["data_loader"]["train"]["args"]
    slam = Ramp_vo(cfg=cfg_VO, network=network, train_cfg=train_cfg)
    for t, (image, events, intrinsics, mask) in enumerate(
        tqdm(_data_iterator(data_list))
    ):
        image, events = resize_input(image, events)
        slam(t, input_tensor=(events, image, mask), intrinsics=intrinsics)

    for _ in range(12):
        slam.update()
        
    points = slam.points_.cpu().numpy()[:slam.m]
    colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
    poses, tstamps = slam.terminate()
    return poses, tstamps, points, colors


def evaluate_sequence(
    config_VO, net, eval_cfg, data_list, traj_ref, use_pose_pred, img_timestamps,
):
    if use_pose_pred:
        # Tune starting_t_to_pred and t_horizon_to_pred accordingly for your dataset
        starting_t_to_pred = traj_ref.num_poses // 2
        t_horizon_to_pred = traj_ref.num_poses - starting_t_to_pred
        
        traj_est, tstamps = run_pose_pred(
            cfg_VO=config_VO,
            network=net,
            eval_cfg=eval_cfg,
            data_list=data_list,
            t_to_pred=starting_t_to_pred,
            t_horizon_to_pred=t_horizon_to_pred,
            deg_approx=4,
        )
    else:
        traj_est, tstamps, points, colors = run(
            cfg_VO=config_VO, network=net, eval_cfg=eval_cfg, data_list=data_list
        )

    traj_est_ = PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:][:, (1, 2, 3, 0)],
        timestamps=img_timestamps,
    )
    
    save_output_for_COLMAP("colmap_saving", traj_est_, points, colors, fx, fy, cx, cy)


    try:
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est_)
        result = main_ape.ape(
            traj_ref=traj_ref,
            traj_est=traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
        )
        ate_score = result.stats["rmse"]
        rot_score = rot_error_with_alignment_from_pose3d(
            ref=traj_ref, est=traj_est, correct_scale=True
        )
    except:
        ate_score = 1000
        rot_score = [1000, 1000, 1000]

    return ate_score, rot_score, traj_est, traj_ref


@torch.no_grad()
def evaluate(
    net, trials=1, downsample_fact=1, config_VO=None, eval_cfg=None, results_path=None
):
    test_ = eval_cfg["data_loader"]["test"]
    train_ = eval_cfg["data_loader"]["train"]["args"]
    norm_to = train_["norm_to"] if train_.get("norm_to") else None
    test_split = test_["test_split"]
    dataset_name = test_["dataset_name"]
    use_pose_pred = test_["use_pose_pred"]

    if config_VO is None:
        config_VO = VO_cfg
        config_VO.merge_from_file("config/default.yaml")

    results = {}
    for scene in test_split:
        print(f"loading training data ... scene:{scene}")
        if not os.path.exists(scene):
            raise FileNotFoundError(f"scene {scene} not found")
        traj_ref_path = osp.join(scene, "pose_left.txt")
        scene_name = os.path.basename(scene) if os.path.isdir(scene) else scene
        timestamps_path = osp.join(scene, "timestamps.txt")
        img_timestamps = np.loadtxt(timestamps_path)

        if "Tartan" in dataset_name:
            set_global_params(K_path=osp.join(scene, "K.yaml"))
            traj_ref = read_tartan_format_poses(
                traj_path=traj_ref_path, timestamps_path=timestamps_path
            )
        elif "StereoDavis" in dataset_name:
            set_global_params(
                K_path=osp.join(scene, "K.yaml"),
                standard_pose_format=True,
            )
            img_timestamps = img_timestamps / 1e6
            traj_ref = read_stereodavis_format_poses(
                traj_path=osp.join(scene, "poses.txt"),
                timestamps_path=osp.join(scene, "timestamps_poses.txt"),
            )
        elif "EDS" in dataset_name:
            set_global_params(
                K_path=osp.join(scene, "K.yaml"),
                standard_pose_format=True,
            )
            img_timestamps = img_timestamps / 1e6
            traj_ref = read_eds_format_poses(traj_ref_path)
        elif "MoonLanding" in dataset_name:
            set_global_params(K_path=osp.join(scene, "K.yaml"))
            traj_ref = read_moonlanding_format_poses(
                traj_path=traj_ref_path, timestamps_path=timestamps_path
            )
        else:
            raise NotImplementedError("dataset not supported")

        data_list, frame_indices = data_loader_all_events(
            config=eval_cfg,
            full_scene=scene,
            downsample_fact=downsample_fact,
            norm_to=norm_to,
        )

        eval_subtraj = partial(
            evaluate_sequence,
            config_VO=config_VO,
            net=net,
            eval_cfg=eval_cfg,
            data_list=data_list,
            traj_ref=traj_ref,
            use_pose_pred=use_pose_pred,
            img_timestamps=img_timestamps[frame_indices],
        )
        save_res = partial(save_results, scene=scene_name, eval_type="full_data")

        results[scene] = {}
        for j in range(trials):
            ate_error, rot_error, traj_est, traj_ref = eval_subtraj()
            print("\n full_data ate ------->", ate_error)
            print("\n full_data rot ------->", rot_error)
            save_res(traj_est=traj_est, traj_ref=traj_ref, j=j)
            results[scene][f"trial_{j}"] = {
                "ate": ate_error,
                "rot_err": list(rot_error),
            }

        if results_path is not None:
            with open(results_path, "w") as json_file:
                json.dump(results, json_file, indent=4)

    if results_path is not None:
        with open(results_path, "w") as json_file:
            results["test_info"] = [
                {"config_VO": dict(config_VO)},
                train_,
                test_,
            ]
            json.dump(results, json_file, indent=4)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="dpvo.pth")
    parser.add_argument("--config_VO", default="config/default.yaml")
    parser.add_argument("--config_eval", type=str, default="config/TartanEvent.json")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--downsample_fact", type=int, default=1)
    parser.add_argument("--results_path", type=str, default=None)

    args = parser.parse_args()

    VO_cfg.merge_from_file(args.config_VO)
    eval_cfg = json.load(open(args.config_eval))

    print("Running evaluation...")

    results = evaluate(
        config_VO=VO_cfg,
        eval_cfg=eval_cfg,
        net=args.weights,
        trials=args.trials,
        downsample_fact=args.downsample_fact,
        results_path=args.results_path,
    )
    for k in results:
        print(k, results[k])
