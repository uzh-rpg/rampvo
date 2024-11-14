from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from torchvision.transforms import Grayscale
from torch.nn import CosineSimilarity
import torch.nn.functional as F
import torch.distributions as D
import matplotlib.pyplot as plt
from .lietorch import SE3
from pathlib import Path
from tqdm import tqdm 
from . import altcorr
import numpy as np
import torch
import time
import os

from data import H5EventHandle


all_times = []
class Timer:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()

    def __exit__(self, type, value, traceback):
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed = self.start.elapsed_time(self.end)
            all_times.append(elapsed)
            print(self.name, elapsed)


def coords_grid(b, n, h, w, **kwargs):
    """coordinate grid"""
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)
    coords = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    return coords[[1, 0]].view(1, 1, 2, h, w).repeat(b, n, 1, 1, 1)


def coords_grid_with_index(d, **kwargs):
    """coordinate grid with frame index"""
    b, n, h, w = d.shape
    i = torch.ones_like(d)
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    coords = torch.stack([x, y, d], dim=2)
    index = torch.arange(0, n, dtype=torch.float, **kwargs)
    index = index.view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index


def patchify(x, patch_size=3):
    """extract patches from video"""
    b, n, c, h, w = x.shape
    x = x.view(b * n, c, h, w)
    y = F.unfold(x, patch_size)
    y = y.transpose(1, 2)
    return y.reshape(b, -1, c, patch_size, patch_size)


def pyramidify(fmap, lvls=[1]):
    """turn fmap into a pyramid"""
    b, n, c, h, w = fmap.shape

    pyramid = []
    for lvl in lvls:
        gmap = F.avg_pool2d(fmap.view(b * n, c, h, w), lvl, stride=lvl)
        pyramid += [gmap.view(b, n, c, h // lvl, w // lvl)]

    return pyramid


def all_pairs_exclusive(n, **kwargs):
    ii, jj = torch.meshgrid(torch.arange(n, **kwargs), torch.arange(n, **kwargs))
    k = ii != jj
    return ii[k].reshape(-1), jj[k].reshape(-1)


def set_depth(patches, depth):
    patches[..., 2, :, :] = depth[..., None, None]
    return patches


def flatmeshgrid(*args, **kwargs):
    grid = torch.meshgrid(*args, **kwargs)
    return (x.reshape(-1) for x in grid)


def image_gradient(images):
    gray = ((images + 0.5) * (255.0 / 2)).sum(
        dim=2
    )  # sum over channel dim -> grayscale
    dx = gray[..., :-1, 1:] - gray[..., :-1, :-1]
    dy = gray[..., 1:, :-1] - gray[..., :-1, :-1]
    g = torch.sqrt(dx**2 + dy**2)
    g = F.avg_pool2d(g, 4, 4)
    return g


def get_coords_gradient_biased(images, patches_per_image, h, w, n):
    g = image_gradient(images)
    x = torch.randint(1, w - 1, size=[n, 3 * patches_per_image], device="cuda")
    y = torch.randint(1, h - 1, size=[n, 3 * patches_per_image], device="cuda")

    coords = torch.stack([x, y], dim=-1).float()
    g = altcorr.patchify(g, coords, 0).view(-1)

    ix = torch.argsort(g)
    x = torch.gather(x, 1, ix[:, -patches_per_image:])
    y = torch.gather(y, 1, ix[:, -patches_per_image:])
    coords = torch.stack([x, y], dim=-1).float()
    return coords


def get_coords_events_biased(events, patches_per_image):
    positive_event_tensor = torch.abs(events.squeeze(0))
    downsampled_event_tensor = F.avg_pool2d(positive_event_tensor.squeeze(0), 4, 4)

    event_in_xy_form = downsampled_event_tensor.transpose(3, 2)
    non_zero_ev = torch.nonzero(torch.mean(event_in_xy_form, dim=1))
    unique_image_index = non_zero_ev[:, 0].unique()

    per_img_nn_zero_ev = [
        non_zero_ev[non_zero_ev[:, 0] == val] for val in unique_image_index
    ]

    # TODO torch.multinomial
    ev_perm = [
        elem[torch.randperm(elem.size()[0])][:patches_per_image, 1:]
        for elem in per_img_nn_zero_ev
    ]

    coords = torch.stack(ev_perm).float()
    return coords


def nms_image(image_tensor, kernel_size=3):
    """
    Performs non-maximum suppression on each channel of a 3D tensor representing an image.

    Args:
    - image_tensor: torch.Tensor of shape (C, H, W)
    - kernel_size: int, size of non maximum suppression around maximums

    Returns:
    - out_tensor: torch.Tensor of shape (C, H, W), float tensor, suppressed version of image_tensor
    """

    image_tensor = image_tensor.unsqueeze(0)
    padding = (kernel_size - 1) // 2

    # Max pool over height and width dimensions
    max_vals = torch.nn.functional.max_pool2d(
        image_tensor, kernel_size, stride=1, padding=padding
    )
    max_vals = max_vals.squeeze(0)

    # Perform non-maximum suppression
    mask = max_vals == image_tensor
    mask = mask.squeeze(0)
    image_tensor = image_tensor.squeeze(0)

    return image_tensor * mask.float()


def get_coords_from_topk_events(
    events,
    patches_per_image,
    border_suppression_size=0,
    non_max_supp_rad=0,
):
    positive_event_tensor = torch.abs(events.squeeze(0))
    downsampled_event_tensor = F.avg_pool2d(positive_event_tensor, 4, 4)
    event_in_xy_form = downsampled_event_tensor.transpose(3, 2)
    ev_mean = torch.mean(event_in_xy_form, dim=1)

    if border_suppression_size != 0:
        # set the borders to 0
        ev_mean[:, :border_suppression_size, :] = 0
        ev_mean[:, -border_suppression_size:, :] = 0
        ev_mean[:, :, :border_suppression_size] = 0
        ev_mean[:, :, -border_suppression_size:] = 0

    if non_max_supp_rad != 0:
        # perform non maximum suppression
        ev_mean = nms_image(ev_mean, kernel_size=non_max_supp_rad)

    event_mean_flat = torch.flatten(ev_mean, start_dim=1, end_dim=-1)
    values, indices = torch.topk(event_mean_flat, k=patches_per_image, dim=-1)

    # compute the row and column indices of the top k values in the flattened tensor
    row_indices = indices / ev_mean.shape[-1]
    col_indices = indices % ev_mean.shape[-1]

    # compute the batch indices of the top k values in the flattened tensor
    batch_indices = (
        torch.arange(ev_mean.shape[0], device="cuda")
        .view(-1, 1)
        .repeat(1, patches_per_image)
    )

    # combine the batch, row, and column indices to obtain the indices in the original 3D tensor
    orig_indices = torch.stack((batch_indices, row_indices, col_indices), dim=-1)

    coords = orig_indices[:, :, 1:]
    return coords


def check_input_tensors(events, images):
    # events.shape = torch.Size([1, 15, 5, 480, 640]) (TRAIN)
    # events.shape = torch.Size([1, 1, 5, 480, 640]) (EVAL)
    if  not (len(events.shape) == len(images.shape)):
        raise AssertionError("Event and image tensor must have \
                             the same number of dimension")
    if not len(events.shape) == 5 and len(images.shape) == 5:
        raise AssertionError("Event and image tensor must have \
                             shape [batch, n_tensors, channels, height, width]")
    if not (events.shape[0] == 1 and images.shape[0] == 1):
        raise NotImplementedError("Event and image tensor must have \
                                  batch dimension (0 dim) = 1. \
                                  Multiple batches not yet implemented")
    


def get_channel_dim(cfg):
    return (cfg["num_event_bins"], 3)



def preprocess_input(input_tensor):
    if len(input_tensor) < 3:
        events, images = input_tensor 
    elif len(input_tensor) == 3:
        events, images, mask = input_tensor
    check_input_tensors(events, images)
    return (events, images, mask)


def stream_from_data(events, images):
    """Generate non batched events and images stream as they arrive
    """
    stream = []
    for frame_ind, (event, image) in enumerate(zip(events.squeeze(0), images.squeeze(0))):
        stream.append({"data": event[None,None,...], "type": "events", "frame_ind": frame_ind })
        stream.append({"data": image[None,None,...], "type": "image", "frame_ind": frame_ind })
    return stream


def apply_loss_to_lstm_states(lstm_states, Loss_func):
    short_term_memo_loss, long_term_memo_loss = [], []
    for state_pair in lstm_states:
        events_state, image_state = state_pair
        hidden_ev, cell_value_ev = events_state
        hidden_img, cell_value_img = image_state
        short_term_memo_loss.append(Loss_func(hidden_ev, hidden_img))
        long_term_memo_loss.append(Loss_func(cell_value_ev, cell_value_img)) 
    return torch.stack(short_term_memo_loss), torch.stack(long_term_memo_loss)

def apply_loss_to_lstm_values(lstm_values, Loss_func):
    loss = []
    for value_pair in lstm_values:
        events_val, image_val = value_pair
        unnormalized_loss = Loss_func(events_val, image_val)
        normalized_loss = unnormalized_loss/len(events_val)
        loss.append(normalized_loss)
    return torch.stack(loss)

def kl_divergence(embedding1, embedding2):
    # compute kl divergence between two 3D arrays
    # the distribution is 2D Normal one for each channel
    channel_dimension = embedding1.shape[1]
    embedding1 = embedding1.reshape(embedding1.shape[1], -1)
    embedding2 = embedding2.reshape(embedding2.shape[1], -1)

    # Compute mean and covariance matrix for each embedding
    mean1 = torch.mean(embedding1, dim=-1)
    cov1 = torch.cov(embedding1)

    mean2 = torch.mean(embedding2, dim=-1)
    cov2 = torch.cov(embedding2)

    # Compute KL divergence between two Gaussian distributions
    dist1 = D.MultivariateNormal(mean1, cov1)
    dist2 = D.MultivariateNormal(mean2, cov2)
    kl_div = torch.distributions.kl.kl_divergence(dist1, dist2)

    #normalized_kl_div = 1 - torch.exp(-(kl_div/channel_dimension) * 0.01)
    #normalized_kl_div = kl_div/channel_dimension

    return kl_div


def cross_entropy(embedding1, embedding2):
    embedding1 = embedding1.reshape(embedding1.shape[1], -1)
    embedding2 = embedding2.reshape(embedding2.shape[1], -1)

    similarity_scores = torch.matmul(embedding1, embedding2.transpose(0, 1))

    # Apply softmax to convert the similarity scores into probabilities
    probabilities = F.softmax(similarity_scores, dim=1)

    # Compute the cross-entropy loss between the predicted probabilities and the target probabilities
    batch_size = probabilities.shape[0]

    # Uniform distribution over all possible pairs of embeddings
    target_probabilities = torch.ones((batch_size, batch_size), device="cuda") / batch_size
    loss = F.binary_cross_entropy(probabilities, target_probabilities)
    return loss

def positive_cos_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(embedding1.shape[1], -1)
    embedding2 = embedding2.reshape(embedding2.shape[1], -1)
    Cos = CosineSimilarity(dim=0)
    # cosine distance is between -1 and +1 
    similarity_scores = 1-Cos(x1=embedding1, x2=embedding2)
    # objective is maximize cos distance
    return torch.mean(similarity_scores)


def precompute_event_indices(event_path, timestamps_path, num_events, indices_file):
    if not Path(event_path).is_file():
        print("ERROR: no event file found")
    event = H5EventHandle.from_path(event_path)
    image_timestamps = np.genfromtxt(timestamps_path)
    i1 = event.find_index_from_timestamp(image_timestamps)
    i0 = np.clip(i1 - num_events, 0, len(event) - 1)
    np.savetxt(indices_file,(i0, i1),delimiter=",",)


def precompute_all_indices(scenes, num_events):
    for segment in tqdm(scenes):
        event_file = Path(scenes[segment]["events"])
        timestamps_path = Path(segment + "/timestamps.txt")
        indices_file = Path(segment + "/indices.txt")
        if indices_file.is_file():
            continue
        precompute_event_indices(
            event_path=event_file,
            timestamps_path=timestamps_path,
            num_events=num_events,
            indices_file=indices_file,
        )


def timer(last_t, time_training=False, section_name="0"):
    if time_training:
        torch.cuda.synchronize()
        current_time = time.time()
        elapsed = int((current_time - last_t) * 1000)
        print(f"section {section_name} elapsed time = {elapsed} ms")
        return current_time
    return 0


def custom_collate(batch):
    # Convert the batch of tuples into a tuple of batches
    batch = list(zip(*batch))
    
    # Collate each batch separately
    collated = []
    for data in batch:
        if isinstance(data[0], torch.Tensor):
            collated.append(torch.stack(data))
        else:
            collated.append(data)
    
    return tuple(collated)

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def pad_input(input_, desired_height, desired_width):
    # Calculate the amount of padding required
    h_diff = desired_height - input_.shape[-2]
    w_diff = desired_width - input_.shape[-1]

    padding = (w_diff // 2, w_diff - w_diff // 2, h_diff // 2, h_diff - h_diff // 2)

    # Pad the image
    return F.pad(input_, padding, mode='constant', value=0)

def input_resize(image, events, desired_ht=480, desired_wh=640):
    if image.shape[-2] == desired_ht and image.shape[-1] == desired_wh:
        return image, events
    
    # TODO padding may confuse the network
    if image.shape[-2] > desired_ht or image.shape[-1] > desired_wh:
        image = F.interpolate(image.squeeze(0), size=(desired_ht, desired_wh), mode='bilinear', align_corners=False)
        image.unsqueeze_(0)
        events = F.interpolate(events.squeeze(0), size=(desired_ht, desired_wh), mode='bilinear', align_corners=False)
        events.unsqueeze_(0)
        return image, events
    
    if image.shape[-2] < desired_ht or image.shape[-1] < desired_wh:
        image = pad_input(input_=image, desired_height=480, desired_width=640)
        events = pad_input(input_=events, desired_height=480, desired_width=640)
        return image, events
    
    raise NotImplementedError("This should not happen")

def initialize_current_pose(last_poses, MOTION_MODEL, MOTION_DAMPING):
    if MOTION_MODEL == 'DAMPED_LINEAR':
        P1 = SE3(last_poses[-1])
        P2 = SE3(last_poses[-2])
        
        xi = MOTION_DAMPING * (P1 * P2.inv()).log()
        tvec_qvec = (SE3.exp(xi) * P1).data
        return tvec_qvec
    else:
        tvec_qvec = last_poses[-1]
        return tvec_qvec

def grayscale_image(image):
    grayscale_image = Grayscale()(image)
    return torch.cat((grayscale_image,grayscale_image,grayscale_image), dim=-3)


def area_under_curve(errors, th_start=0.05, th_stop=1, th_num=20, return_auc_by_threshold=False):
    thresholds = np.linspace(start=th_start, stop=th_stop, num=th_num, endpoint=True)[...,None]
    errors = np.array(errors).T
    assert len(errors.shape) == 2
    assert errors.shape[0] == 1
    diff_thresholded = np.maximum((thresholds-errors),0)
    if th_start != th_stop:
        diff_thresholded[diff_thresholded>0]=1
        
    if return_auc_by_threshold:
        return diff_thresholded.mean(axis=-1)
    return diff_thresholded.mean()


def average_results(results):
    full = []
    for scene in results:
        full_med = np.median(results[scene])
        full.append(full_med)
    return full

def min_results(results):
    full = []
    for scene in results:
        full_med = np.nanmin(results[scene])
        full.append(full_med)
    return full


# Visualizations scripts for extrinsic to pyramid visualizer

def get_visualizer(traj):
    scale_fac = -10
    if traj.shape[-1] == 4:
        positions_xyz = traj[:, 0:3, 3]
    else:
        positions_xyz = traj[:, :3]
    # visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [0, 10])
    visualizer = CameraPoseVisualizer(
        [min(positions_xyz[:, 0] - scale_fac), max(positions_xyz[:, 0] + scale_fac)],
        [min(positions_xyz[:, 1] - scale_fac), max(positions_xyz[:, 1] + scale_fac)],
        [min(positions_xyz[:, 2] - scale_fac), max(positions_xyz[:, 2] + scale_fac)],
    )
    return visualizer


def get_extrinsic(traj, timestamps):
    positions_xyz = traj[:, :3]
    orientations_quat_wxyz = traj[:, 3:]
    extrinsic = PoseTrajectory3D(
        positions_xyz=positions_xyz,
        orientations_quat_wxyz=orientations_quat_wxyz,
        timestamps=timestamps,
    )
    return extrinsic


def visualize_camera_trajectory(
    traj_est, timestamps, traj_name="default_traj", traj_ref=None
):
    plt.clf()
    visualizer_est = get_visualizer(traj_est)
    extrinsic_est = get_extrinsic(traj_est, timestamps)
    T_est = extrinsic_est.poses_se3

    for frame_id, elem in enumerate(T_est):
        if frame_id % 2 != 0:
            continue
        visualizer_est.extrinsic2pyramid(
            elem,
            plt.cm.rainbow(frame_id / len(T_est)),
            focal_len_scaled=0.1,
            aspect_ratio=0.5,
        )
    plt.show()
    path = "trajectory_visualization/" + traj_name
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + ".png")

    input("Press Enter to continue visualization...")

    if traj_ref is not None:
        plt.clf()
        visualizer_ref = get_visualizer(traj_ref)
        extrinsic_ref = get_extrinsic(traj_ref, timestamps)
        # alignment_transformation = lie_algebra.sim3(*extrinsic_est.align(extrinsic_ref, correct_only_scale=True))

        T_ref = extrinsic_ref.poses_se3
        # T_ref_scaled = np.multiply(T_ref, alignment_transformation)
        # visualizer_ref = get_visualizer(T_ref_scaled)

        for frame_id, elem in enumerate(T_est):
            if frame_id % 10 != 0:
                continue
            visualizer_ref.extrinsic2pyramid(
                T_ref[frame_id], color="g", focal_len_scaled=0.1, aspect_ratio=0.3,
            )

        plt.show()
        gt_name = os.path.join(
            os.path.dirname(traj_name), os.path.basename(traj_name) + "_gt"
        )
        path = "trajectory_visualization/" + gt_name
        plt.savefig(path + ".png")

    input("Press Enter to continue visualization...")


def filter_features(confidences, target, data_shape):
    ht, wd = data_shape

    filter_binary = torch.ones_like(target, dtype=torch.int)
    x_coordinates = target[:, :, 0]
    y_coordinates = target[:, :, 1]

    x_mask = torch.logical_or(x_coordinates < 0, x_coordinates > wd)
    y_mask = torch.logical_or(y_coordinates < 0, y_coordinates > ht)

    final_mask = torch.logical_or(x_mask, y_mask)
    filter_binary[final_mask] = 0

    return confidences * filter_binary
    

def normalize_image(images, norm_img_to):
    # TODO if the range is not 0-255 if overflows the specified range
    if norm_img_to == "-1_1":
        images = 2 * (images / 255.0) - 1
        #transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Apply standard pytorch transformation to the image
        #images = transform(images)
    else:
        # normalize between -0.5 and 1.5
        images = 2 * (images / 255.0) - 0.5
    return images


def interpolate_poses(poses, target_timestamps, original_timestamps):
    interpolated_trajectory = []
    for target_time in target_timestamps:
        # Step 2: Identify the relevant data points
        index_before = np.searchsorted(original_timestamps, target_time) - 1
        index_after = index_before + 1

        # handle edge case of target time be after the last t or before the first t of original t stamps
        if index_after >= len(original_timestamps):
            interpolated_trajectory.append(poses[index_before])
            continue
        if index_before < 0:
            interpolated_trajectory.append(poses[index_after])
            continue

        # Step 3: Calculate interpolation factors
        time_before = original_timestamps[index_before]
        time_after = original_timestamps[index_after]
        alpha = (target_time - time_before) / (time_after - time_before)

        x_before, y_before, z_before, qx_before, qy_before, qz_before, qw_before = poses[index_before]
        x_after, y_after, z_after, qx_after, qy_after, qz_after, qw_after = poses[index_after]

        # Step 4: Perform linear interpolation for position (x, y, z)
        x_interpolated = x_before + alpha * (x_after - x_before)
        y_interpolated = y_before + alpha * (y_after - y_before)
        z_interpolated = z_before + alpha * (z_after - z_before)

        # Step 4: Perform linear interpolation for quaternion (qx, qy, qz, qw)
        R_before = Rotation.from_quat([qx_before, qy_before, qz_before, qw_before]).as_matrix()
        R_after = Rotation.from_quat([qx_after, qy_after, qz_after, qw_after]).as_matrix()
        key_rots = Rotation.from_matrix(np.stack((R_before, R_after), axis=0))
        key_times = [time_before, time_after]

        # Step 5: Evaluate the interpolated pose
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(target_time)
        q_interpolated = interp_rots.as_quat()
        interpolated_pose = (x_interpolated, y_interpolated, z_interpolated, *q_interpolated)

        #print("Interpolated Pose at time {}: {}".format(target_time, interpolated_pose))
        interpolated_trajectory.append(interpolated_pose)

    return np.stack(interpolated_trajectory, axis=0)



def save_output_for_COLMAP(name: str, traj: PoseTrajectory3D, points: np.ndarray, colors: np.ndarray, fx, fy, cx, cy, H=480, W=640):
    """ Saves the sparse point cloud and camera poses such that it can be opened in COLMAP """

    colmap_dir = Path(name)
    colmap_dir.mkdir(exist_ok=True)
    scale = 10 # for visualization

    # images
    images = ""
    traj = PoseTrajectory3D(poses_se3=list(traj.poses_se3), timestamps=traj.timestamps)
    for idx, (x,y,z), (qw, qx, qy, qz) in zip(range(1,traj.num_poses+1), traj.positions_xyz*scale, traj.orientations_quat_wxyz):
        images += f"{idx} {qw} {qx} {qy} {qz} {x} {y} {z} 1\n\n"
    (colmap_dir / "images.txt").write_text(images)

    # points
    points3D = ""
    colors_uint = (colors * 255).astype(np.uint8).tolist()
    for i, (p,c) in enumerate(zip((points*scale).tolist(), colors_uint), start=1):
        points3D += f"{i} " + ' '.join(map(str, p + c)) + " 0.0 0 0 0 0 0 0\n"
    (colmap_dir / "points3D.txt").write_text(points3D)

    # camera
    (colmap_dir / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}")
    print(f"Saved COLMAP-compatible reconstruction in {colmap_dir.resolve()}")
