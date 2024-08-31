import numpy as np
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from evo.core import sync
from evo.tools import plot
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D

def quaternion_rotate_xz_only(quaternion):
    # Create a Rotation object from the quaternion
    r = R.from_quat(quaternion)
    
    # Convert the rotation to Euler angles (in radians)
    euler = r.as_euler('xyz')  # Get Euler angles (x, y, z)
    
    # Set the rotation around the y axis to zero
    euler[1] = 0  # Set rotation around y to zero
    
    # Create a new Rotation object from Euler angles
    r_new = R.from_euler('xyz', euler)
    
    # Convert the new rotation to quaternion
    new_quaternion = r_new.as_quat()
    
    return new_quaternion

def compute_euler_rpy(trajectory):
    eulers = []
    for pose in trajectory:
        quaternion = pose[4:]
        r = R.from_quat(quaternion)
        euler = r.as_euler('xyz')
        eulers.append(euler)
    eulers_ = np.stack(eulers)
    return eulers_

def plot_each_angle(est, ref, axis=0):
    plt.clf()
    est_euler = compute_euler_rpy(est)
    ref_euler = compute_euler_rpy(ref)
    plt.plot(est_euler[:,axis], label="Est")
    plt.plot(ref_euler[:,axis], label="Ref")
    plt.show()
    plt.savefig(f"/home/pellerito/Automatic_dataset_conversion/euler_align_axis_{axis}.png")


def trajectory_with_angles(trajectory):
    eulers = compute_euler_rpy(trajectory)
    timestamps = trajectory[:,0]
    trajectory_ = PoseTrajectory3D(
                positions_xyz=eulers,
                orientations_quat_wxyz=trajectory[:,3:][:, (1,2,3,0)],
                timestamps=timestamps)
    return trajectory_

def align_trajectories(est, ref):
    est_ = trajectory_with_angles(est)
    ref_ = trajectory_with_angles(ref)
    result = main_ape.ape(ref_, est_, est_name='traj', pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    print(result.stats["rmse"]*360/np.pi)




def compute_euler_rpy_from_quat(quaternions):
    eulers = []
    for quaternion in quaternions:
        r = R.from_quat(quaternion)
        euler = r.as_euler('xyz')
        eulers.append(euler)
    eulers_ = np.stack(eulers)
    return eulers_

def trajectory_with_angles_from_pose3d(trajectory: PoseTrajectory3D):
    quaternions_xyzw = trajectory.orientations_quat_wxyz[:, (1,2,3,0)]
    eulers = compute_euler_rpy_from_quat(quaternions_xyzw)
    timestamps = trajectory.timestamps
    trajectory_ = PoseTrajectory3D(
                positions_xyz=eulers,
                orientations_quat_wxyz=quaternions_xyzw[:, (1,2,3,0)],
                timestamps=timestamps)
    return trajectory_

def select_trajectory_subset(trajectory_, start, stop):
    trajectory = deepcopy(trajectory_)
    pos = trajectory.positions_xyz[start:stop, :]
    quat = trajectory.orientations_quat_wxyz[start:stop, :]
    timestamps = trajectory.timestamps[start:stop]

    return PoseTrajectory3D(
            positions_xyz=pos,
            orientations_quat_wxyz=quat,
            timestamps=timestamps)

def angle_error(theta1, theta2):
    if np.any(theta1 > np.pi) or np.any(theta1 < -np.pi) or np.any(theta2 > np.pi) or np.any(theta2 < -np.pi):
        print("Angles must be in the range [-pi, pi]")
        return None
        # raise ValueError("Angles must be in the range [-pi, pi]")
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

def rot_error_with_alignment_from_pose3d(est: PoseTrajectory3D, ref: PoseTrajectory3D, correct_scale):
    """
    Compute the rotation error in degree between two trajectories after aligning euler angles with Umeyama aligment
    """
    est_ = trajectory_with_angles_from_pose3d(deepcopy(est))
    ref_ = trajectory_with_angles_from_pose3d(deepcopy(ref))
    
    # estimate in place change
    est_.align(ref_, correct_scale, False, n=-1)
    angles_est = est_.positions_xyz
    angles_ref = ref_.positions_xyz

    errors = []
    for angle1, angle2 in zip (angles_est, angles_ref):
        angle_error_ = angle_error(angle1, angle2)
        if angle_error_ is not None:
            errors.append(angle_error_)
    errors_ = np.stack(errors)
    angle_error_degree = np.rad2deg(np.mean(np.abs(errors_)))
    angle_error_degree_per_axis = np.rad2deg(np.mean(np.abs(errors_), axis=0))
    return angle_error_degree_per_axis