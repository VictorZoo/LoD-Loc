# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat
import numpy as np
from .utils import deg2rad, rotmat2d


def location_error(uv, uv_gt, ppm=1):
    return torch.norm(uv - uv_gt.to(uv), dim=-1)

def location_error_single(uv, uv_gt, ppm=1):
    return abs(uv - uv_gt.to(uv))

def angle_error(t, t_gt):
    error = torch.abs(t % 360 - t_gt.to(t) % 360)
    error = torch.minimum(error, 360 - error)
    return error

def error_4x4(GPS_pose, GT_pose):
    e_t = torch.norm(GPS_pose[...,:3,3] - GT_pose[...,:3,3], dim=1)
    batch = GPS_pose.shape[0]
    e_R_ = np.zeros(batch)
    for i in range(batch):
        cos = np.clip((np.trace(np.dot(GPS_pose[i, :3,:3].cpu().numpy(), GT_pose[i, :3,:3].cpu().numpy().T)) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))
        e_R_[i] = e_R
    return e_t, torch.tensor(e_R_).to(e_t.device)

    
class Location2DRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold,  key="uv_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = location_error(pred[self.key][...,3:], data["pose_GT"][...,3:]) #XYZ
        super().update((error <= self.threshold).float())

class AllRecall(torchmetrics.MeanMetric):
    def __init__(self, xyz_threshold, angle_threshold, key="rxyz_max", *args, **kwargs):
        self.angle_threshold = angle_threshold
        self.xyz_threshold = xyz_threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        e_t, e_R = error_4x4(pred[self.key], data["pose_GT_4x4"].to(pred[self.key]))
        # xyz_error = location_error(pred[self.key][..., 3:], data["pose_GT"][...,3:])
        flag = torch.logical_and(e_R <= self.angle_threshold, e_t <= self.xyz_threshold)
        super().update(flag.float())

class Location2DRecall_xy(torchmetrics.MeanMetric):
    def __init__(self, threshold,  key="uv_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = location_error(pred[self.key][...,3:-1], data["pose_GT"][...,3:-1]) #XYZ
        super().update((error <= self.threshold).float())

class Location2DRecall_z(torchmetrics.MeanMetric):
    def __init__(self, threshold,  key="uv_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = location_error_single(pred[self.key][...,-1], data["pose_GT"][...,-1]) #XYZ
        super().update((error <= self.threshold).float())

class AngleRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="yaw_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = angle_error(pred[self.key][..., 2], data["pose_GT"][..., 2])
        super().update((error <= self.threshold).float())


class MeanMetricWithRecall(torchmetrics.Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("value", default=[], dist_reduce_fx="cat")

    def compute(self):
        return dim_zero_cat(self.value).mean(0)

    def get_errors(self):
        return dim_zero_cat(self.value)

    def recall(self, thresholds):
        error = self.get_errors()
        thresholds = error.new_tensor(thresholds)
        return (error.unsqueeze(-1) < thresholds).float().mean(0) * 100


class AngleError(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        value = angle_error(pred[self.key][...,2], data["pose_GT"][..., 2])
        if value.numel():
            self.value.append(value)


class Location2DError(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def update(self, pred, data):

        value = location_error(pred[self.key][...,3:], data["pose_GT"][...,3:])
        if value.numel():
            self.value.append(value)

class Location2DError_x(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key
        # self.ppm = pixel_per_meter

    def update(self, pred, data):
        value = location_error_single(pred[self.key][...,3], data["pose_GT"][...,3])
        # print("Location error: ", value)
        if value.numel():
            self.value.append(value)

class Location2DError_y(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key
        # self.ppm = pixel_per_meter

    def update(self, pred, data):
        value = location_error_single(pred[self.key][...,4], data["pose_GT"][...,4])
        # print("Location error: ", value)
        if value.numel():
            self.value.append(value)

class Location2DError_z(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key
        # self.ppm = pixel_per_meter

    def update(self, pred, data):
        value = location_error_single(pred[self.key][...,5], data["pose_GT"][...,5])
        # print("Location error: ", value)
        if value.numel():
            self.value.append(value)


class LateralLongitudinalError(MeanMetricWithRecall):
    def __init__(self, pixel_per_meter, key="uv_max"):
        super().__init__()
        self.ppm = pixel_per_meter
        self.key = key

    def update(self, pred, data):
        yaw = deg2rad(data["roll_pitch_yaw"][..., -1])
        shift = (pred[self.key] - data["uv"]) * yaw.new_tensor([-1, 1])
        shift = (rotmat2d(yaw) @ shift.unsqueeze(-1)).squeeze(-1)
        error = torch.abs(shift) / self.ppm
        value = error.view(-1, 2)
        if value.numel():
            self.value.append(value)
