from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation
from sklearn import gaussian_process
from collections import defaultdict
import matplotlib.pyplot as plt
from ramp.lietorch import SE3
import numpy as np
import torch
from ramp.utils import flatmeshgrid


def compute_relative_pose_error(pose1, pose2):
    """
    Compute the relative pose error between two poses.

    Args:
        pose1 (np.ndarray): The first pose as a 1x7 numpy array [x, y, z, quaternion_x, quaternion_y, quaternion_z, quaternion_w].
        pose2 (np.ndarray): The second pose as a 1x7 numpy array [x, y, z, quaternion_x, quaternion_y, quaternion_z, quaternion_w].

    Returns:
        np.ndarray: The relative pose error as a 6D vector [tx, ty, tz, rx, ry, rz].

    """
    # Extract translation vectors from poses
    t1 = pose1[:3]
    t2 = pose2[:3]

    # Extract quaternions from poses
    q1 = pose1[3:]
    q2 = pose2[3:]

    # Normalize quaternions
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    # Convert quaternions to rotation matrices
    R1 = Rotation.from_quat([q1[3], q1[0], q1[1], q1[2]]).as_matrix()
    R2 = Rotation.from_quat([q2[3], q2[0], q2[1], q2[2]]).as_matrix()

    # Compute translation error
    translation_error = ((t2 - t1) ** 2).sum() ** 0.5

    # Compute rotation error as Euler angles
    rotation_error = Rotation.from_matrix(R1.T @ R2).as_euler('xyz', degrees=True)
    rotation_error = (rotation_error ** 2).sum() ** 0.5 # Square to get mean squared error 

    return translation_error, rotation_error


# function to compute relative pose error between two single poses
def relative_pose_error(pose1, pose2):
    pose1 = np.ndarray(pose1)
    pose2 = np.ndarray(pose2)
    rel_pose = SE3(pose1).inv() * SE3(pose2)
    return rel_pose.log().norm().item()


# Plot utils for debugging and computing flows for patch prediction
def plot_flow_from_start_image(start_img, images_list, image_mapping, patch_dict, ii, jj, kk, name):
    for next_image in set(jj[ii==start_img].tolist()):
        start_rgb_image = images_list[image_mapping[start_img]]
        end_rgb_image = images_list[image_mapping[next_image]]
        patch_range=kk[(ii==start_img) & (jj==next_image)]

        b,n,c,h,w = images_list[0].shape 
        if c > 3:
            rescale_factor = 1
        else:
            rescale_factor = 4

        print_single_img_flow(
            start_img=start_img, 
            end_image=next_image, 
            start_rgb_image=start_rgb_image, 
            end_rgb_image=end_rgb_image,
            patch_range=patch_range.cpu().numpy(),
            rescale_factor=rescale_factor, 
            patch_dict=patch_dict, 
            name=name
            )


def print_single_img_flow(start_img, end_image, start_rgb_image, end_rgb_image, patch_dict, patch_range, rescale_factor=4, name="."):
    fig, ax = plt.subplots(1, 1, figsize=(20, 9))
    
    start_rgb_image = start_rgb_image.cpu().detach()[0,0,...].permute(1,2,0)
    end_rgb_image = end_rgb_image.cpu().detach()[0,0,...].permute(1,2,0)
    combined_image = np.concatenate((start_rgb_image, end_rgb_image), axis=1)
    height, width, _ = end_rgb_image.shape

    ax.axis([ -100, 2*width+100, height+100, -100])

    ax.imshow(combined_image)
    for patch_ind in patch_range:
        x0, y0 = patch_dict[start_img, start_img, patch_ind].cpu().T * rescale_factor
        x1, y1 = patch_dict[start_img, end_image, patch_ind].cpu().T * rescale_factor
        x1 = x1 + width
        flow = (x1-x0, y1-y0)

        color = "g"
        if y1 > height or y1 < 0:
            color = "r"
        if x1 > 2*width or x1 < 0:
            color = "r"

        ax.arrow(x=x0, y=y0, dx=flow[0], dy=flow[1], color=color, width=3, length_includes_head=True)
    
    number_string = "%0*d" % (3, end_image)
    main_name = f"DEBUG/flow_{name}_from_{start_img}_to_{number_string}"
    plt.savefig(main_name)



def compute_image_to_patch_map(coords, ii, jj, kk, to_gpu=False):
    patch_dict = {}
    if not to_gpu:
        ii = ii.cpu().numpy()
        jj = jj.cpu().numpy()
        kk = kk.cpu().numpy()
    for start_image, end_image, patch_id in zip(ii,jj,kk):
        mask = (ii==start_image) & (jj==end_image) & (kk==patch_id)
        patch = coords[0,mask,:,0,0]
        if len(patch) == 0:
            continue
        patch_dict[start_image, end_image, patch_id] = patch
    return patch_dict


def compute_patch_track(coords, ii, jj, kk, image_to_proj=-1, to_gpu=False):
    if not to_gpu:
        ii = ii.cpu().numpy()
        jj = jj.cpu().numpy()
        kk = kk.cpu().numpy()
    patch_dict = defaultdict(list)
    for start_image, end_image, patch_id in zip(ii,jj,kk):
        if ((ii==start_image) & (jj==image_to_proj)).any() == False:
            continue
        
        mask = (ii==start_image) & (kk==patch_id)
        patch = coords[0,mask,:,0,0]
        if len(patch) == 0 or len(patch_dict[start_image, patch_id]) > 0:
            continue

        patch_dict[start_image, patch_id] = patch
    return patch_dict


def compute_patch_track_(coords, ii, jj, kk, image_to_proj=-1, to_gpu=False):
    if not to_gpu:
        ii = ii.cpu().numpy()
        jj = jj.cpu().numpy()
        kk = kk.cpu().numpy()
    patch_dict = defaultdict(list)
    for start_image, end_image, patch_id in zip(ii,jj,kk):
        if image_to_proj>0 and ((ii==start_image) & (jj==image_to_proj)).any() == False:
            continue
        
        mask = (ii==start_image) & (jj==end_image) & (kk==patch_id)
        patch = coords[0,mask,:,0,0]
        if len(patch) == 0:
            continue

        patch_dict[start_image, patch_id].append(patch)
    return patch_dict


def compute_patch_track__(coords, ii, jj, kk, image_to_proj):
    patch_dict = defaultdict(list)
    new_image_mask = (jj==image_to_proj)
    start_edge = ii[new_image_mask]
    patch_edge = kk[new_image_mask]

    for start_image, patch_id in zip(start_edge,patch_edge):
        start_image = start_image.cpu().item()
        patch_id = patch_id.cpu().item()
        if ((ii==start_image) & (jj==image_to_proj)).any() == False:
            continue
        
        mask = (ii==start_image) & (kk==patch_id)
        patch = coords[0,mask,:,0,0]
        if len(patch) == 0 or len(patch_dict[start_image, patch_id]) > 0:
            continue
        
        patch_dict[start_image, patch_id] = patch
    return patch_dict


def motion_bootstrap(n, poses, MOTION_MODEL, MOTION_DAMPING):
    if MOTION_MODEL == 'DAMPED_LINEAR':
        P1 = SE3(poses[n-1])
        P2 = SE3(poses[n-2])
        
        xi = MOTION_DAMPING * (P1 * P2.inv()).log()
        tvec_qvec = (SE3.exp(xi) * P1).data
        return tvec_qvec
    else:
        return poses[n-1]
    

def add_forward_elements(frame_num, patch_extracted_num, r, ii, jj, kk, ix, weights):
    t0 = patch_extracted_num * max((frame_num - r), 0)
    t1 = patch_extracted_num * max((frame_num - 1), 0)
    kk_toadd, jj_toadd = flatmeshgrid(
        torch.arange(t0, t1, device="cuda"), torch.arange(frame_num-1, frame_num, device="cuda"), indexing='ij')
    
    ii_stack = torch.cat([ii, ix[kk_toadd]])
    jj_stack = torch.cat([jj, jj_toadd])
    kk_stack = torch.cat([kk, kk_toadd])

    new_weights = torch.zeros((1, len(kk_toadd), 2), device="cuda")
    weights_stack = torch.cat([weights, new_weights], dim=1)

    return ii_stack, jj_stack, kk_stack, weights_stack


# for each starting image, once the starting image is fixed there is a patch on the target image when the target image is changed
# which is the same, so if we know that the last frame, for example 32 is connected with keyframes 20, 21, 22, ... 31, we can use all patches
# the patch tracks of each of these frames (20, 21, 22, ... 31) to predict their projection in frame 32
def predict_patch_pos(step_to_pred_future,
                           next_frame_index, 
                           coords,
                           weights, 
                           patch_dict, 
                           img_to_keyframe_map, 
                           ii, jj, kk, 
                           data_shape, 
                           frequency=30, deg=2,
                           gp_prediction=False,
                           ):
    steps = 1
    past_patch_num = 5
    height, width = data_shape
    # patch_id: is the unique id extracted in each frame, in frame 0 ids=0:95, in frame 1 ids=96:191
    for start_patch_pair in patch_dict.keys():
        start_image, patch_id = start_patch_pair
        first_connected_frame = jj[ii==start_image].min()

        # discard coords reprojection on the next virtual frame
        x,y = patch_dict[start_patch_pair][:-1].T.cpu().numpy()
        t = (img_to_keyframe_map[first_connected_frame:next_frame_index] / frequency).cpu().numpy()

        x_mask = (x>=0) & (x<width)
        y_mask = (y>=0) & (y<height)
        mask = x_mask & y_mask

        if np.all(mask[-past_patch_num:]==False):
            masked_weights = 0
        else:
            masked_weights = 10**-9

        x_ = x[-past_patch_num:]
        y_ = y[-past_patch_num:]
        t_ = t[-past_patch_num:]
        w = (t_-t_[0])/(t[-1]-t_[0]) + 10**-7
        assert len(t_) == len(x_)
        spl_x = UnivariateSpline(x=t_, y=x_, w=w, bbox=[None, None], k=deg, s=None, ext=0, check_finite=False)
        spl_y = UnivariateSpline(x=t_, y=y_, w=w, bbox=[None, None], k=deg, s=None, ext=0, check_finite=False)

        new_time = t[-1]+(step_to_pred_future/frequency)
        new_x = torch.tensor(spl_x(new_time))
        new_y = torch.tensor(spl_y(new_time))

        # produce a grid of 3x3 around the predicted point
        x = torch.arange(new_x - steps, new_x + steps + 1)[:3]
        y = torch.arange(new_y - steps, new_y + steps + 1)[:3]
        cols_grid, rows_grid = torch.meshgrid(x, y)

        mask = (ii==start_image) & (kk==patch_id)
        edge_mask = mask & (jj==next_frame_index)

        coords[:,edge_mask,:,:,:] = torch.stack((rows_grid,cols_grid), dim=0).cuda()
        weights[:,edge_mask,:] = masked_weights

    return coords.cuda(), weights.cuda()


def fit_model_patch_track(next_frame_index, 
                           patch_dict, 
                           img_to_keyframe_map, 
                           ii, jj, 
                           data_shape, 
                           frequency=30, deg=2):
    past_patch_num = 5
    height, width = data_shape
    patch_models = {}
    # patch_id: is the unique id extracted in each frame, in frame 0 ids=0:95, in frame 1 ids=96:191
    for start_patch_pair in patch_dict.keys():
        start_image, patch_id = start_patch_pair
        first_connected_frame = jj[ii==start_image].min()

        # discard coords reprojection on the next virtual frame
        x,y = patch_dict[start_patch_pair][:-1].T.cpu().numpy()
        t = (img_to_keyframe_map[first_connected_frame:next_frame_index] / frequency).cpu().numpy()

        x_mask = (x>=0) & (x<width)
        y_mask = (y>=0) & (y<height)
        mask = x_mask & y_mask

        #if mask[-past_patch_num:].all() == False:
        if np.all(mask[-past_patch_num:]==False):
            masked_weights = 0
        else:
            masked_weights = 10**-9

        x_ = x[-past_patch_num:]
        y_ = y[-past_patch_num:]
        t_ = t[-past_patch_num:]
        w = (t_-t_[0])/(t[-1]-t_[0]) + 10**-7
        assert len(t_) == len(x_)
        spl_x = UnivariateSpline(x=t_, y=x_, w=w, bbox=[None, None], k=deg, s=None, ext=0, check_finite=False)
        spl_y = UnivariateSpline(x=t_, y=y_, w=w, bbox=[None, None], k=deg, s=None, ext=0, check_finite=False)

        last_t_stamp = t_[-1]
        patch_models[start_patch_pair] = (spl_x, spl_y, masked_weights, last_t_stamp)
        
    return patch_models


def predict_patch_on_model(patch_models, 
                           step_to_pred_future, 
                           frequency, 
                           next_frame_index, 
                           coords, weights, 
                           ii, jj, kk):
    steps=1
    for start_patch_pair in patch_models.keys():
        start_image, patch_id = start_patch_pair
        spl_x, spl_y, masked_weights, last_t_stamp = patch_models[start_patch_pair]

        new_time = last_t_stamp+(step_to_pred_future/frequency)
        new_x = torch.tensor(spl_x(new_time))
        new_y = torch.tensor(spl_y(new_time))

        # produce a grid of 3x3 around the predicted point
        x = torch.arange(new_x - steps, new_x + steps + 1)[:3]
        y = torch.arange(new_y - steps, new_y + steps + 1)[:3]
        cols_grid, rows_grid = torch.meshgrid(x, y)

        mask = (ii==start_image) & (kk==patch_id)
        edge_mask = mask & (jj==next_frame_index)

        coords[:,edge_mask,:,:,:] = torch.stack((rows_grid,cols_grid), dim=0).cuda()
        weights[:,edge_mask,:] = masked_weights

    return coords.cuda(), weights.cuda()