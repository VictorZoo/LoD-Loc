# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

import numpy as np
from scipy.spatial.transform import Rotation
import torch

def qvec2rotmat(qvec):  #!wxyz
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def crop_map(raster, xy, size, seed=None):
    h, w = raster.shape[-2:]
    state = np.random.RandomState(seed)
    top = state.randint(0, h - size + 1)
    left = state.randint(0, w - size + 1)
    raster = raster[..., top : top + size, left : left + size]
    xy -= np.array([left, top])
    return raster, xy

def parse_pose_list(path):
    all_pose_c2w = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            q, t = np.split(np.array(data[1:], float), [4])
            
            R = np.asmatrix(qvec2rotmat(q)).transpose()   #c2w
            Pose_c2w = np.identity(4)
            Pose_c2w[0:3,0:3] = R
            Pose_c2w[0:3, 3] = -R.dot(t) #t  ##!!! 改

            all_pose_c2w.append(Pose_c2w)
    return np.array(all_pose_c2w)

def point_proj (sampled_points_array, rt_4x4_batch, K, origin_corr, w, h, num_sample):
    # sampled_points_array: (num_points, 3)
    # rt_4x4_batch: (N, 4, 4)   C2W
    # K: Camera intrinsic matrix  W2C
    # origin_corr: Origin correction vector
    # w: Width
    # h: Height

    N = rt_4x4_batch.shape[0]

    # Expand sampled_points_array to N dimensions
    sampled_points_array = sampled_points_array.unsqueeze(0).expand(N, -1, -1).cuda()

    homogeneous_points = torch.cat((sampled_points_array, torch.ones((N, sampled_points_array.shape[1], 1), dtype=torch.float64).cuda()), dim=-1)

    # Convert origin_corr to torch tensor
    origin_corr_tensor = origin_corr.cuda()

    rt_4x4_batch[:, :3, 3] -= origin_corr_tensor  # Update translation components for each batch
    rt_4x4_batch_inv = torch.inverse(rt_4x4_batch)  # Compute the inverse transformation matrix
    transformed_points = torch.matmul(rt_4x4_batch_inv, homogeneous_points.permute(0, 2, 1))  # Apply inverse transformation

    projected_points = torch.matmul(K, transformed_points)
    projected_points = projected_points.permute(0, 2, 1)
    pixel_coordinates = projected_points[:, :, :2] / projected_points[:, :, 2:]

    # Clamp pixel_coordinates to be within [0, w] and [0, h]
    pixel_coordinates[:, :, 0] = torch.clamp(pixel_coordinates[:, :, 0], 0, w)
    pixel_coordinates[:, :, 1] = torch.clamp(pixel_coordinates[:, :, 1], 0, h)

    pixel_Cz = projected_points[:, :, 2:]


    return pixel_coordinates, pixel_Cz

def read_intrinsic(intrinsc_path):
    all_K = []
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:]))[:]
            K_w2c = np.array([
            [fx,0.0,cx,0],
            [0.0,fy,cy,0],
            [0.0,0.0,1.0,0],
            ]) 
            all_K.append(K_w2c)
    
    return np.array(all_K), w,h


def random_rot90(raster, xy, heading, seed=None):
    rot = np.random.RandomState(seed).randint(0, 4)
    heading = (heading + rot * np.pi / 2) % (2 * np.pi)
    h, w = raster.shape[-2:]
    if rot == 0:
        xy2 = xy
    elif rot == 2:
        xy2 = np.array([w, h]) - 1 - xy
    elif rot == 1:
        xy2 = np.array([xy[1], w - 1 - xy[0]])
    elif rot == 3:
        xy2 = np.array([h - 1 - xy[1], xy[0]])
    else:
        raise ValueError(rot)
    raster = np.rot90(raster, rot, axes=(-2, -1))
    return raster, xy2, heading


def random_flip(image, raster, xy, heading, seed=None):
    state = np.random.RandomState(seed)
    if state.rand() > 0.5:  # no flip
        return image, raster, xy, heading
    image = image[:, ::-1]
    h, w = raster.shape[-2:]
    if state.rand() > 0.5:  # flip x
        raster = raster[..., :, ::-1]
        xy = np.array([w - 1 - xy[0], xy[1]])
        heading = np.pi - heading
    else:  # flip y
        raster = raster[..., ::-1, :]
        xy = np.array([xy[0], h - 1 - xy[1]])
        heading = -heading
    heading = heading % (2 * np.pi)
    return image, raster, xy, heading


def decompose_rotmat(R_c2w):
    R_cv2xyz = Rotation.from_euler("X", -90, degrees=True)
    rot_w2c = R_cv2xyz * Rotation.from_matrix(R_c2w).inv()
    roll, pitch, yaw = rot_w2c.as_euler("YXZ", degrees=True)
    return roll, pitch, yaw
