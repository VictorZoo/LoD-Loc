# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

import math
import os
from typing import Optional
from scipy.spatial.transform import Rotation as R
import torch
from functools import reduce
import torch.nn.functional as F
from .voting import  get_score, loss_rxyz
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

transf = torch.tensor([
                    [1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1.],
                ])

def trans_eulerTo4x4(combined_samples):
    degree, xyz = combined_samples[...,:3], combined_samples[...,3:]
    n, _ = degree.shape
    ret_2 = R.from_euler('xyz', degree, degrees=True)
    R_ = ret_2.as_matrix()
    T = torch.eye(4)
    T = T.unsqueeze(0).expand(n, -1, -1).clone()
    T[:,0:3,0:3] = torch.from_numpy(R_)
    T[:,0:3,3] = xyz
    T = T @ torch.inverse(transf.to(T.dtype))# C2W

    return T

def trans_W2C(pred_R, pred_t):
    b = pred_R.shape[0]
    device = pred_R.device
    pred_R = pred_R.transpose(1,2)
    pred_t = -pred_R @ pred_t.unsqueeze(2)
    T = torch.eye(4)
    T = T.unsqueeze(0).expand(b, -1, -1).clone()
    T[:,0:3,0:3] = pred_R
    T[:,0:3,3] = pred_t.squeeze()
    xyz, euler = pose2euler(T)
    rxyz = torch.hstack([euler[...,2].unsqueeze(1), xyz])
    return T.to(device), rxyz.to(device)

def visualize_feature_map(feature_map, save_pth):
    """
    将特征图保存为图像文件

    Parameters:
    - feature_map: 二维数组，表示特征图
    - file_path: 要保存的文件路径，包括文件名和扩展名（例如：'output.png'）

    Returns:
    无返回值，将特征图保存到指定路径
    """
    normalized_map = F.normalize(feature_map, p=2, dim=(0, 1))
    # breakpoint()
    # normalized_map[normalized_map < -0.0014] = 0
    plt.imshow(normalized_map, cmap='viridis')  # 使用viridis颜色映射，也可以选择其他颜色映射
    plt.axis('off')
    # plt.colorbar()  # 添加颜色条
    # plt.title('Feature Map Visualization')
    plt.savefig(save_pth, dpi=300, bbox_inches='tight', pad_inches=0)

def multi_stage_loss(pred, gt_rxyz, loss_weight, num_stage):
    tot_loss = 0.
    success = None

    for stage_id in range(num_stage): #3
        ranges = pred["stage{}".format(stage_id+1)]['ranges']
        log_prob = pred["stage{}".format(stage_id+1)]['log_prob']
        sample_rxyz = pred["stage{}".format(stage_id+1)]['sample_euler']
        num_sample = pred["stage{}".format(stage_id+1)]["num_sample"]
        loss, success = loss_rxyz(log_prob, gt_rxyz, sample_rxyz, ranges, num_sample)
    
        tot_loss += loss * loss_weight[stage_id]
        
    return tot_loss

def multi_stage_loss_l1loss(pred, gt_rxyz, loss_weight, num_stage):
    tot_loss = 0.
    for stage_id in range(num_stage): 
        rxyz_pred = pred["stage{}".format(stage_id+1)]['rxyz_pred']
        loss = F.smooth_l1_loss(rxyz_pred, gt_rxyz, reduction='mean')
        tot_loss += loss * loss_weight[stage_id]
        
    return tot_loss

def multi_stage_loss_KL(pred, data, loss_weight, num_stage):
    tot_loss = 0.

    for stage_id in range(num_stage):
        
        pred_score = pred["stage{}".format(stage_id+1)]['pred_score']
        
        f_weight = pred["stage{}".format(stage_id+1)]["w_feature"]
        # data['pose_sample'] 应该是GT的pose
        
        uv_gt = point_proj(data["points3D"], data['pose_sample'], data['intrinsic'], data['origin_hw'])
        _, _, new_h, new_w = f_weight.shape
        scale_factors = torch.stack([(new_h-1) / data['origin_hw'][:, 0], (new_w-1) / data['origin_hw'][:, 1]], dim=1)

        uv_gt[:, :, :, 0] *= scale_factors[0,1]
        uv_gt[:, :, :, 1] *= scale_factors[0,0]
        uv_gt = norm_uv(uv_gt, new_h, new_w)
        gt_score = get_score(f_weight, uv_gt)
        gt_score = gt_score.squeeze()
        
        input = F.log_softmax(pred_score, dim=-1)
        target = F.softmax(gt_score, dim=-1)

        loss = F.kl_div(input, target, reduction='sum')
        tot_loss += loss * loss_weight[stage_id]
        
    return tot_loss

def find_max(score_mean, poses_sampled, score):
    b = poses_sampled.shape[0]
    batch_indices = torch.arange(b)
    max_score_mean, max_indices = torch.max(score_mean, dim=1)
    max_poses = poses_sampled[batch_indices, max_indices]
    max_score = score[batch_indices, max_indices, ...]
    return max_score_mean, max_poses , max_score

def pose2euler(pose):
    device = pose.device
    pose = pose.cpu() @ transf
    xyz , R_ = pose[:,0:3,3], pose[:,0:3,0:3]
    ret_init = R.from_matrix(R_)
    initial_euler = torch.from_numpy(ret_init.as_euler('xyz',degrees=True))
    return xyz.to(device), initial_euler.to(device) #Pitch Roll Yaw

def euler2pose(degree, xyz):
    # degree, xyz = [...,:3], combined_samples[...,3:]
    n, _ = degree.shape
    device = degree.device
    degree = degree.cpu()
    xyz = xyz.cpu()

    ret_2 = R.from_euler('xyz', degree, degrees=True)
    R_ = ret_2.as_matrix()
    T = torch.eye(4)
    T = T.unsqueeze(0).expand(n, -1, -1).clone()
    T[:,0:3,0:3] = torch.from_numpy(R_)
    T[:,0:3,3] = xyz
    T = T @ torch.inverse(transf.to(T.dtype))# C2W
    return T.to(device)

def plot_points_on_image(image_path, points, output_path='output_image.jpg'):
    """
    在彩色图像上绘制二维点，并保存结果。

    参数:
    - image_path (str): 输入彩色图像文件的路径。
    - points (numpy.ndarray): 二维点的坐标，形状为 (N, 2)。
    - output_path (str): 输出图像的文件路径，默认为 'output_image.jpg'。
    """

    # 读取彩色图像
    image = Image.open(image_path)

    # 创建图像
    plt.figure(figsize=(10, 8))
    plt.imshow(image)  # 显示彩色图像

    # 绘制二维点
    plt.scatter(points[:, 0], points[:, 1], color='red', marker='o', s=0.1, label='Points')
 
    # 设置坐标轴范围
    plt.xlim(0, image.width)
    plt.ylim(image.height, 0)  # 注意y轴的方向

    # 显示图例
    plt.legend()

    # 保存带有点的图像
    plt.savefig(output_path)
    
    plt.cla()
    plt.close("all")
    # plt.show()  # 显示图像



def sample_poses(initial_xyz, initial_R, Yawxyz_error_ranges, num_samples_Yawxyz, pose_GT):
    if (len(Yawxyz_error_ranges.shape)) == 2:
        Yawxyz_error_ranges = Yawxyz_error_ranges.unsqueeze(0).expand(initial_xyz.shape[0],-1,-1)
   
    # Yawxyz_error_ranges = Yawxyz_error_ranges.to(initial_xyz.dtype)
    device = initial_xyz.device
    initial_xyz = initial_xyz.cpu()
    initial_R = initial_R.cpu()
    b, _ = initial_xyz.shape
    num_sample = reduce(lambda x, y: x * y, num_samples_Yawxyz)
    stacked_tensor = torch.empty((0,num_sample,4,4))
    stacked_euler = torch.empty((0,num_sample,6))

    ret_init = R.from_matrix(initial_R)
    initial_euler = torch.from_numpy(ret_init.as_euler('xyz',degrees=True))
    pitch, roll, initial_yaw = torch.unbind(initial_euler, dim=-1)
    
    
    sample_euler = torch.cat([initial_euler,initial_xyz], dim = 1)
    # 在误差范围内均匀采样
    yaw_error_range, xyz_error_ranges = Yawxyz_error_ranges[:,0], Yawxyz_error_ranges[:,1:]
    num_samples_yaw, num_samples_xyz = num_samples_Yawxyz[0], num_samples_Yawxyz[1:]
    # 在batch循环采样，
    for j in range(b):
        xyz_samples = [
            torch.linspace(
                initial_xyz[j,i] + float(xyz_error_ranges[j,i,0]),
                initial_xyz[j,i] + float(xyz_error_ranges[j,i,1]),
                num_samples_xyz[i]
            ) for i in range(3)]
        yaw_samples = torch.linspace(initial_yaw[j] + float(yaw_error_range[j,0]), initial_yaw[j] + float(yaw_error_range[j,1]), num_samples_yaw)
        
        pitch = pitch.to(yaw_samples[0].dtype)
        roll = roll.to(yaw_samples[0].dtype)

        pitch_samples, roll_samples, yaw_samples, xyz_samples_0, xyz_samples_1, xyz_samples_2 = torch.meshgrid(pitch[j], roll[j], yaw_samples, *xyz_samples)

        combined_samples = torch.stack([pitch_samples, roll_samples, yaw_samples, xyz_samples_0, xyz_samples_1, xyz_samples_2], dim=-1)
        combined_samples = combined_samples.squeeze()

        T_ = trans_eulerTo4x4(combined_samples.view(-1, 6))
        T_ = T_.unsqueeze(0)
        stacked_tensor = torch.cat([stacked_tensor, T_], dim=0)
        stacked_euler = torch.cat([stacked_euler, combined_samples.view(-1, 6).unsqueeze(0)], dim=0)

    return stacked_tensor.to(device), stacked_euler[:,:,2:].to(device), sample_euler[:,2:].to(device), initial_euler[:,:2].to(device)

def sample_poses_sixDoF(initial_xyz, initial_R, Rxyz_error_ranges, num_samples_Yawxyz):
    if (len(Rxyz_error_ranges.shape)) == 2:
        Rxyz_error_ranges = Rxyz_error_ranges.unsqueeze(0).expand(initial_xyz.shape[0],-1,-1)
   
    device = initial_xyz.device
    initial_xyz = initial_xyz.cpu()
    initial_R = initial_R.cpu()
    b, _ = initial_xyz.shape
    num_sample = reduce(lambda x, y: x * y, num_samples_Yawxyz)
    stacked_tensor = torch.empty((0,num_sample,4,4))
    stacked_euler = torch.empty((0,num_sample,6))

    ret_init = R.from_matrix(initial_R)
    initial_euler = torch.from_numpy(ret_init.as_euler('xyz',degrees=True))
    pitch, roll, initial_yaw = torch.unbind(initial_euler, dim=-1)
    sample_euler = torch.cat([initial_euler,initial_xyz], dim = 1)
    # 在误差范围内均匀采样
    r_error_range, xyz_error_ranges = Rxyz_error_ranges[:,:3], Rxyz_error_ranges[:,3:]
    num_samples_yaw, num_samples_xyz = num_samples_Yawxyz[:3], num_samples_Yawxyz[3:]
    # 在batch循环采样，
    for j in range(b):
        xyz_samples = [
            torch.linspace(
                initial_xyz[j,i] + float(xyz_error_ranges[j,i,0]),
                initial_xyz[j,i] + float(xyz_error_ranges[j,i,1]),
                num_samples_xyz[i]
            ) for i in range(3)]
        
        r_samples = [
            torch.linspace(
                initial_euler[j,i] + float(r_error_range[j,i,0]),
                initial_euler[j,i] + float(r_error_range[j,i,1]),
                num_samples_yaw[i]
            ) for i in range(3)]

        pitch_samples, roll_samples, yaw_samples, xyz_samples_0, xyz_samples_1, xyz_samples_2 = torch.meshgrid(*r_samples, *xyz_samples)
        combined_samples = torch.stack([pitch_samples, roll_samples, yaw_samples, xyz_samples_0, xyz_samples_1, xyz_samples_2], dim=-1)
        combined_samples = combined_samples.squeeze()

        T_ = trans_eulerTo4x4(combined_samples.view(-1, 6))
        T_ = T_.unsqueeze(0)
        stacked_tensor = torch.cat([stacked_tensor, T_], dim=0)
        stacked_euler = torch.cat([stacked_euler, combined_samples.view(-1, 6).unsqueeze(0)], dim=0)
    return stacked_tensor.to(device), stacked_euler.to(device), sample_euler.to(device), initial_euler[:,:2].to(device)

def point_proj (sampled_points_array, rt_4x4_batch, K, hw):
    # sampled_points_array: ( N, num_points, 3)
    # rt_4x4_batch: (N, num_points,  4, 4)  C2W
    # K: Camera intrinsic matrix W2C
    # w: Width
    # h: Height

    if len(rt_4x4_batch.shape) == 3:
        rt_4x4_batch = rt_4x4_batch.unsqueeze(1).to(sampled_points_array.dtype)
    K = K.to(sampled_points_array.dtype)
    N, num_sample = rt_4x4_batch.shape[:2]
    h, w = hw[0,0], hw[0,1]
    # Expand sampled_points_array to N dimensions

    # sampled_points_array = sampled_points_array.unsqueeze(0).expand(N, -1, -1).cuda()

    homogeneous_points = torch.cat((sampled_points_array, torch.ones((N, sampled_points_array.shape[1], 1), dtype=sampled_points_array.dtype).to(sampled_points_array)), dim=-1)
    homogeneous_points = homogeneous_points.unsqueeze(1).expand(-1, num_sample, -1, -1)
    K = K.unsqueeze(1).expand(-1, num_sample, -1, -1)

    rt_4x4_batch_inv = torch.inverse(rt_4x4_batch)  # Compute the inverse transformation matrix

    transformed_points = torch.matmul(rt_4x4_batch_inv, homogeneous_points.permute(0, 1, 3, 2))  # Apply inverse transformation

    projected_points = torch.matmul(K, transformed_points)
    del transformed_points
    del homogeneous_points
    projected_points = projected_points.permute(0, 1, 3, 2)
    pixel_coordinates = projected_points[:, :, :, :2] / projected_points[:, :, :, 2:]

    # Clamp pixel_coordinates to be within [0, w] and [0, h]
    pixel_coordinates[:, :, :, 0] = torch.clamp(pixel_coordinates[:, :, :, 0], 0, w-1)
    pixel_coordinates[:, :, :, 1] = torch.clamp(pixel_coordinates[:, :, :, 1], 0, h-1)


    return pixel_coordinates

def norm_uv(uv_sampled, new_h, new_w):
    uv_sampled[:, :, :, 0] = (uv_sampled[:, :, :, 0]/(new_w-1)) * 2 -1
    uv_sampled[:, :, :, 1] = (uv_sampled[:, :, :, 1]/(new_h-1)) * 2 -1
    return uv_sampled

def interpolate_points(pos, depth):

    B, C, N, _ = pos.shape
    ids = torch.arange(0, B)
    
    _, _, h, w = depth.size()
    depth = depth.squeeze()
    i = pos[:, :, :, 0]
    j = pos[:, :, :, 1]

    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    # 插值出来的深度
    ids = ids.unsqueeze(1).unsqueeze(2)
    interpolated_depth = (
        w_top_left * depth[ids[:], i_top_left, j_top_left].view(B, C, -1) +
        w_top_right * depth[ids[:], i_top_right, j_top_right].view(B, C, -1) +
        w_bottom_left * depth[ids[:], i_bottom_left, j_bottom_left].view(B, C, -1) +
        w_bottom_right * depth[ids[:], i_bottom_right, j_bottom_right].view(B, C, -1)
    )
    return interpolated_depth#.cuda()

def checkpointed(cls, do=True):
    """Adapted from the DISK implementation of Michał Tyszkiewicz."""
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls


class GlobalPooling(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "mean":
            self.fn = torch.nn.Sequential(
                torch.nn.Flatten(2), torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten()
            )
        elif kind == "max":
            self.fn = torch.nn.Sequential(
                torch.nn.Flatten(2), torch.nn.AdaptiveMaxPool1d(1), torch.nn.Flatten()
            )
        else:
            raise ValueError(f"Unknown pooling type {kind}.")

    def forward(self, x):
        return self.fn(x)


@torch.jit.script
def make_grid(
    w: float,
    h: float,
    step_x: float = 1.0,
    step_y: float = 1.0,
    orig_x: float = 0,
    orig_y: float = 0,
    y_up: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    x, y = torch.meshgrid(
        [
            torch.arange(orig_x, w + orig_x, step_x, device=device),
            torch.arange(orig_y, h + orig_y, step_y, device=device),
        ],
        # indexing="xy",
    )
    if y_up:
        y = y.flip(-2)
    grid = torch.stack((x, y), -1)
    return grid


@torch.jit.script
def rotmat2d(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([c, -s, s, c], -1).reshape(angle.shape + (2, 2))
    return R


@torch.jit.script
def rotmat2d_grad(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([-s, -c, c, -s], -1).reshape(angle.shape + (2, 2))
    return R


def deg2rad(x):
    return x * math.pi / 180


def rad2deg(x):
    return x * 180 / math.pi
