# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

from __future__ import barry_as_FLUFL
from multiprocessing.sharedctypes import Value
from typing import Optional, Tuple
# import torch.nn.functional as F
import numpy as np
import torch
from torch.nn.functional import grid_sample, log_softmax, pad, softmax

eps=1e-5

def sample_xyr(volume, xy_grid, angle_grid, nearest_for_inf=False):
    # (B, C, H, W, N) to (B, C, H, W, N+1)
    volume_padded = pad(volume, [0, 1, 0, 0, 0, 0], mode="circular")

    size = xy_grid.new_tensor(volume.shape[-3:-1][::-1])
    xy_norm = xy_grid / (size - 1)  # align_corners=True
    angle_norm = (angle_grid / 360) % 1
    grid = torch.concat([angle_norm.unsqueeze(-1), xy_norm], -1)
    grid_norm = grid * 2 - 1

    valid = torch.all((grid_norm >= -1) & (grid_norm <= 1), -1)
    value = grid_sample(volume_padded, grid_norm, align_corners=True, mode="bilinear")

    # if one of the values used for linear interpolation is infinite,
    # we fallback to nearest to avoid propagating inf
    if nearest_for_inf:
        value_nearest = grid_sample(
            volume_padded, grid_norm, align_corners=True, mode="nearest"
        )
        value = torch.where(~torch.isfinite(value) & valid, value_nearest, value)
    return value, valid

def calculate_proportions_batch(errors_batch, error_ranges):
    lower_bounds, upper_bounds = zip(*error_ranges)
    lower_bounds = torch.tensor(lower_bounds, dtype=torch.float32, device=errors_batch.device)
    upper_bounds = torch.tensor(upper_bounds, dtype=torch.float32, device=errors_batch.device)
    proportions_batch = (errors_batch - lower_bounds) / (upper_bounds - lower_bounds)
    return proportions_batch.to(dtype=torch.float32)


def sample_xyzr(volume, rxyz_gt, rxyz_pred, error_ranges, num_sample, nearest_for_inf=False):
    r_, x_, y_, z_ = num_sample
    volume = volume.reshape(-1, r_, x_, y_, z_)

    delta_rxyz = rxyz_gt - rxyz_pred

    if len(error_ranges.shape) == 2:
        error_ranges = error_ranges.unsqueeze(0).expand(delta_rxyz.shape[0], -1,-1)
    
    rxyz_norm = torch.empty((0, delta_rxyz.shape[1])).to(delta_rxyz)
    success = torch.empty(0).to(delta_rxyz)
    for i in range(delta_rxyz.shape[0]):
        rxyz_norm_i = calculate_proportions_batch(delta_rxyz[i,:], error_ranges[i,:])
        success_i = torch.all((rxyz_norm_i >= 0.0-eps) & (rxyz_norm_i <= 1.0+eps))
        rxyz_norm = torch.cat([rxyz_norm, rxyz_norm_i.unsqueeze(0)])
        success = torch.cat([success, success_i.unsqueeze(0)])
    rxyz_norm = (rxyz_norm * 2 -1).float() #归一化
    xyz_norm = rxyz_norm[:, 1:]
    r_norm = rxyz_norm[:, 0]
    xyz_norm = xyz_norm.view(-1, 1, 1, 1, 3)[:,:,:,:,[2,1,0]] # 插值的時候xy要反一下？ R X Y Z -> r z y x
    value_i = grid_sample(volume, xyz_norm.to(volume.dtype), align_corners=True, mode="bilinear")
    # breakpoint()
    if value_i.shape[0] == 1:
        value_i = value_i.squeeze(-1).squeeze(-1).squeeze(-1)
        value_i = value_i.unsqueeze(1).unsqueeze(2)
    else:
        value_i = value_i.squeeze()
        value_i = value_i.unsqueeze(1).unsqueeze(2)
    r_norm = torch.stack([r_norm , torch.zeros_like(r_norm)], dim=1)
    r_norm = r_norm.unsqueeze(1).unsqueeze(2)
    
    value = grid_sample(value_i, r_norm.to(value_i.dtype), align_corners=True, mode="bilinear")
    return value, success


def sample_xyzr_6DoF(volume, rxyz_gt, rxyz_pred, error_ranges, num_sample, nearest_for_inf=False):
    pi_, ro_, yaw_, x_, y_, z_ = num_sample
    volume = volume.reshape(-1, pi_, ro_, yaw_, x_, y_, z_)
    delta_rxyz = rxyz_gt - rxyz_pred
    batch = delta_rxyz.shape[0]
    if len(error_ranges.shape) == 2:
        error_ranges = error_ranges.unsqueeze(0).expand(delta_rxyz.shape[0], -1,-1)
    
    rxyz_norm = torch.empty((0, delta_rxyz.shape[1])).to(delta_rxyz)
    success = torch.empty(0).to(delta_rxyz)
    for i in range(delta_rxyz.shape[0]):
        rxyz_norm_i = calculate_proportions_batch(delta_rxyz[i,:], error_ranges[i,:])
        success_i = torch.all((rxyz_norm_i >= 0.0-eps) & (rxyz_norm_i <= 1.0+eps))
        rxyz_norm = torch.cat([rxyz_norm, rxyz_norm_i.unsqueeze(0)])
        success = torch.cat([success, success_i.unsqueeze(0)])
    rxyz_norm = (rxyz_norm * 2 -1).float() #归一化
    xyz_norm = rxyz_norm[:, 3:]
    r_norm = rxyz_norm[:, :3]
    xyz_norm = xyz_norm.view(-1, 1, 1, 1, 3)[:,:,:,:,[2,1,0]] 
    value_i = grid_sample(volume.view(batch,-1, x_, y_, z_), xyz_norm.to(volume.dtype), align_corners=True, mode="bilinear")
    value_i = value_i.reshape(-1, pi_, ro_, yaw_).unsqueeze(1)
    r_norm = r_norm.view(-1, 1, 1, 1, 3)[:,:,:,:,[2,1,0]]
    value = grid_sample(value_i, r_norm.to(value_i.dtype), align_corners=True, mode="bilinear")
    return value, success


def loss_rxyz(log_probs, gt_rxyz, pred_rxyz, error_ranges, num_sample):
    if gt_rxyz.shape[1]==4:
        log_prob, success = sample_xyzr(
        log_probs, gt_rxyz, pred_rxyz, error_ranges, num_sample
    )
    elif gt_rxyz.shape[1]==6:
        log_prob, success = sample_xyzr_6DoF(
        log_probs, gt_rxyz, pred_rxyz, error_ranges, num_sample
    )
    
    nll = -log_prob.reshape(-1) 
    return nll, success


def nll_loss_xyr_smoothed(log_probs, xy, angle, sigma_xy, sigma_r, mask=None):
    *_, nx, ny, nr = log_probs.shape
    grid_x = torch.arange(nx, device=log_probs.device, dtype=torch.float)
    dx = (grid_x - xy[..., None, 0]) / sigma_xy
    grid_y = torch.arange(ny, device=log_probs.device, dtype=torch.float)
    dy = (grid_y - xy[..., None, 1]) / sigma_xy
    dr = (
        torch.arange(0, 360, 360 / nr, device=log_probs.device, dtype=torch.float)
        - angle[..., None]
    ) % 360
    dr = torch.minimum(dr, 360 - dr) / sigma_r
    diff = (
        dx[..., None, :, None] ** 2
        + dy[..., :, None, None] ** 2
        + dr[..., None, None, :] ** 2
    )
    pdf = torch.exp(-diff / 2)
    if mask is not None:
        pdf.masked_fill_(~mask[..., None], 0)
        log_probs = log_probs.masked_fill(~mask[..., None], 0)
    pdf /= pdf.sum((-1, -2, -3), keepdim=True)
    return -torch.sum(pdf * log_probs.to(torch.float), dim=(-1, -2, -3))


def get_score(f_weight, swapped_uv_sampled):

    B_, num_sample, num_points, uv = swapped_uv_sampled.shape
    swapped_uv_sampled= swapped_uv_sampled.reshape(B_, 1, num_sample*num_points, uv)
    grid_f = grid_sample(f_weight, swapped_uv_sampled)
    grid_f = grid_f.squeeze()
    scores = grid_f.reshape(B_, num_sample, num_points)
    return scores

def get_mean_score(scores):
    # 求平均
    scores = scores.mean(dim=(-1))
    return scores

def log_softmax_spatial(x, dims=3):
    return log_softmax(x.flatten(-dims), dim=-1).reshape(x.shape)

def softmax_spatial(x, dims=3):
    return softmax(x.flatten(-dims), dim=-1).reshape(x.shape)


@torch.jit.script
def argmax_xy(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.flatten(-2).max(-1).indices
    width = scores.shape[-1]
    x = indices % width
    y = torch.div(indices, width, rounding_mode="floor")
    return torch.stack((x, y), -1)



@torch.jit.script
def expectation_xyr(
    prob: torch.Tensor, covariance: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    h, w, num_rotations = prob.shape[-3:]
    x, y = torch.meshgrid(
        [
            torch.arange(w, device=prob.device, dtype=prob.dtype),
            torch.arange(h, device=prob.device, dtype=prob.dtype),
        ],
        # indexing="xy",
    )
    grid_xy = torch.stack((x, y), -1)
    xy_mean = torch.einsum("...hwn,hwd->...d", prob, grid_xy)

    angles = torch.arange(0, 1, 1 / num_rotations, device=prob.device, dtype=prob.dtype)
    angles = angles * 2 * np.pi
    grid_cs = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
    cs_mean = torch.einsum("...hwn,nd->...d", prob, grid_cs)
    angle = torch.atan2(cs_mean[..., 1], cs_mean[..., 0])
    angle = (angle * 180 / np.pi) % 360

    if covariance:
        xy_cov = torch.einsum("...hwn,...hwd,...hwk->...dk", prob, grid_xy, grid_xy)
        xy_cov = xy_cov - torch.einsum("...d,...k->...dk", xy_mean, xy_mean)
    else:
        xy_cov = None

    xyr_mean = torch.cat((xy_mean, angle.unsqueeze(-1)), -1)
    return xyr_mean, xy_cov


@torch.jit.script
def argmax_xyr(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.flatten(-3).max(-1).indices
    width, num_rotations = scores.shape[-2:]
    wr = width * num_rotations
    y = torch.div(indices, wr, rounding_mode="floor")
    x = torch.div(indices % wr, num_rotations, rounding_mode="floor")
    angle_index = indices % num_rotations
    angle = angle_index * 360 / num_rotations
    xyr = torch.stack((x, y, angle), -1)
    return xyr



def find_indice_batch(tensor_batch):
    # tensor_batch = torch.tensor(batch_array)
    max_values, max_indices = torch.max(tensor_batch.view(tensor_batch.size(0), -1), 1)
    # 将一维索引转换为四维坐标
    w_size, x_size, y_size, z_size = tensor_batch.size()[1:]
    max_ws = max_indices // (x_size * y_size * z_size)
    remaining = max_indices % (x_size * y_size * z_size)
    max_xs = remaining // (y_size * z_size)
    remaining = remaining % (y_size * z_size)
    max_ys = remaining // z_size
    max_zs = remaining % z_size

    # w->Yaw x->Z y->X z->Y
    # max_values = np.array(max_values.tolist())
    # max_coordinates = np.array(list(zip(max_ws.tolist(), max_xs.tolist(), max_ys.tolist(), max_zs.tolist())))
    max_coordinates = torch.stack([max_ws, max_xs, max_ys, max_zs], dim=1)
    return max_values, max_coordinates

def find_rzxy_batch(error_ranges, num_list, indice, rxyz_gps):
    error_ranges = torch.tensor(error_ranges)[[0,3,1,2],:].to(indice.device)
    lower_bounds, upper_bounds = error_ranges[:, 0], error_ranges[:, 1]
    num_list = torch.tensor(num_list).to(indice.device)
  
    rzxy = (indice * (upper_bounds - lower_bounds) / num_list) + lower_bounds
    rzxy_real = rxyz_gps[:, [0, 3, 1, 2]] + rzxy
    return rzxy_real

