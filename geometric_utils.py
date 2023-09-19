from pyquaternion import Quaternion
import numpy as np
from numba import njit
import pdb
import time

# from: https://github.com/nutonomy/nuscenes-devkit/blob/6d7fc293a092f069efadb0aad2edabc8d8a4f73c/python-sdk/nuscenes/eval/common/utils.py#L112
def quaternion2yaw(quat : Quaternion):
    # """
    # Calculate the yaw angle from a quaternion.
    # Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    # It does not work for a box in the camera frame.
    # :param q: Quaternion of interest.
    # :return: Yaw angle in radians.
    # """
    # # Project into xy plane.
    # v = np.dot(quat.rotation_matrix, np.array([1, 0, 0]))

    # # Measure yaw using arctan.
    # yaw = np.arctan2(v[1], v[0])

    # # convert into degree
    # yaw = yaw / np.pi * 180
    euler = np.array(quat.yaw_pitch_roll) / np.pi * 180

    return euler[0]


def quaternion2roll(quat : Quaternion):
    # v = quat.rotation_matrix

    # # Measure yaw using arctan.
    # roll = np.arctan2(v[2,1], v[2,2])

    # # convert into degree
    # roll = roll / np.pi * 180
    euler = np.array(quat.yaw_pitch_roll) / np.pi * 180

    return euler[2]


def quaternion2pitch(quat : Quaternion):
    # v = quat.rotation_matrix

    # # Measure yaw using arctan.
    # pitch = np.arcsin(v[0,2])

    # # convert into degree
    # pitch = pitch / np.pi * 180
    euler = np.array(quat.yaw_pitch_roll) / np.pi * 180

    return euler[1]


# from: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
def quaternion2rotationmatrix(quat: Quaternion):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    # q0 = Q[0]
    # q1 = Q[1]
    # q2 = Q[2]
    # q3 = Q[3]
     
    # # First row of the rotation matrix
    # r00 = 2 * (q0 * q0 + q1 * q1) - 1
    # r01 = 2 * (q1 * q2 - q0 * q3)
    # r02 = 2 * (q1 * q3 + q0 * q2)
     
    # # Second row of the rotation matrix
    # r10 = 2 * (q1 * q2 + q0 * q3)
    # r11 = 2 * (q0 * q0 + q2 * q2) - 1
    # r12 = 2 * (q2 * q3 - q0 * q1)
     
    # # Third row of the rotation matrix
    # r20 = 2 * (q1 * q3 - q0 * q2)
    # r21 = 2 * (q2 * q3 + q0 * q1)
    # r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # # 3x3 rotation matrix
    # rot_matrix = np.array([[r00, r01, r02],
    #                        [r10, r11, r12],
    #                        [r20, r21, r22]])
                            
    return np.array(quat.rotation_matrix)


def extrinsic_calib(data):
    rot = quaternion2rotationmatrix(Quaternion(data['rotation']))
    ext = np.concatenate([rot, np.array(data['translation']).reshape(-1,1)], axis=1)
    ext = np.concatenate([ext, np.array([0,0,0,1]).reshape(1,-1)])
    return ext


# from: https://github.com/nutonomy/nuscenes-devkit/blob/da3c9a977112fca05413dab4e944d911769385a9/python-sdk/nuscenes/utils/geometry_utils.py#L18
@njit
def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3).reshape(3, nbr_points)

    return points