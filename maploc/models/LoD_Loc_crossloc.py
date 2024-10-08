# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

import time
import numpy as np
import torch
from torch.nn.functional import normalize
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
from . import get_model
from .base import BaseModel
from .submodules import *
from torchvision.utils import save_image
from .utils import(
    point_proj, 
    sample_poses, 
    sample_poses_sixDoF,
    find_max, 
    pose2euler,
    euler2pose,
    multi_stage_loss, 
    multi_stage_loss_l1loss,
    norm_uv, 
    visualize_feature_map,
    multi_stage_loss_KL,
    transf)
from .voting import (
    log_softmax_spatial,
    softmax_spatial,
    get_score,
    get_mean_score,
)
# from .map_encoder import MapEncoder
from .metrics_sixDoF import ( AngleError, AngleRecall, Location2DError_x, Location2DError_y,Location2DError_z,Location2DError, 
                        Location2DRecall_xy, Location2DRecall_z,Location2DRecall,AllRecall )


class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)

class LoD_Loc(BaseModel):
    default_conf = {
        "num_sample": "???",
        "error_ranges" : "???",
        "base_chs": "???",
        "lamb": "???",
        "stage_configs": "???",
        "grad_method": "???",
        "base_chs" : "???",
        "lamb" : "???",
        "feat_ext_ch" : "???",
        "loss_weight" : "???",
        "confidence" : "???",
        "thresh": 0.5,
        "loss_id": "l1_loss",
        "save_pth": "???"
    }

    def _init(self, conf):
        self.stage_configs = self.conf.stage_configs
        self.grad_method = self.conf.grad_method
        self.base_chs = self.conf.base_chs
        self.lamb = self.conf.lamb
        self.error_ranges = self.conf.error_ranges
        self.num_sample = self.conf.num_sample
        self.num_stage = len(self.conf.stage_configs)
        self.loss_weight = self.conf.loss_weight
        self.loss_id = self.conf.loss_id
        self.confidence = self.conf.confidence
        self.save_pth = self.conf.save_pth


        self.feature_extraction = FeatExtNet(base_channels=self.conf.feat_ext_ch, num_stage=self.num_stage)

    def _forward(self, data):
        pred = {}
        features = self.feature_extraction(data["image"])
        exp_var, pred_pose = None, None
        lamb_ = None
        # 根据不同stage进行操作
        for stage_idx in range(self.num_stage): 
            output = {}
            lamb_ = self.lamb[stage_idx]
            features_stage = features["stage{}_f".format(stage_idx + 1)]
            conf_feature = features["stage{}_c".format(stage_idx + 1)]

            if self.confidence:
                f_weight = features_stage  * conf_feature
            else:  
                f_weight = features_stage

            if stage_idx == 0:
                # 初始采样范围
                ranges = torch.tensor(self.error_ranges)
                output["ranges"] = ranges
                output["num_sample"] = torch.tensor(self.num_sample[stage_idx])
                poses_init = data['pose_sample'] @ transf.to(data['pose_sample'])
            else:
                # 根据上一阶段计算的方差来确定range范围
                low_bound = -exp_var
                high_bound = exp_var
                ranges = torch.stack((low_bound, high_bound),dim = 1).permute(0, 2 ,1)
                output["num_sample"] = torch.tensor(self.num_sample[stage_idx])
                output["ranges"] = ranges
                poses_init = pred_pose @ transf.to(pred_pose)

            # Pose采样、投影并取Feature
            poses_sampled, rxyz_sampled, output["sample_euler"], PitchRoll = sample_poses_sixDoF(poses_init[:,0:3,3], poses_init[:,0:3,0:3], ranges, output["num_sample"])
            uv_sampled = point_proj(data["points3D"], poses_sampled, data['intrinsic'], data['origin_hw'])
            _, _, new_h, new_w = f_weight.shape
            scale_factors = torch.stack([(new_h-1) / data['origin_hw'][:, 0], (new_w-1) / data['origin_hw'][:, 1]], dim=1)
            uv_sampled[:, :, :, 0] *= scale_factors[0,1]
            uv_sampled[:, :, :, 1] *= scale_factors[0,0]
            uv_sampled = norm_uv(uv_sampled, new_h, new_w)
            score = get_score(f_weight, uv_sampled) 
            del uv_sampled

            score_mean = get_mean_score(score)
            output["prob"] = softmax_spatial(score_mean, dims = 1)
            
            if self.loss_id == "l1_loss":
                #计算预测的Yaw xyz以及其对应pose
                output["rxyz_pred"] = torch.sum(rxyz_sampled * output["prob"].unsqueeze(2).expand(rxyz_sampled.shape),1)
                with torch.no_grad():
                    bind_eulerPose = output["rxyz_pred"]
                    # bind_eulerPose = torch.cat([PitchRoll, output["rxyz_pred"]],dim=1) # 4DoF
                    output["pred_pose"] = euler2pose(bind_eulerPose[:,:3], bind_eulerPose[:,3:])
                
            ### Softmax Loss 时需要用的步骤
            elif self.loss_id == "softmax" or "kl_loss":
                output["log_prob"] = log_softmax_spatial(score_mean, dims = 1)
                output["w_feature"] = f_weight
                output["pred_score_mean"], output["pred_pose"], output["pred_score"] = find_max(output["log_prob"], poses_sampled, score)
                # output["pred_score_mean"], output["pred_pose"] = find_max(output["prob"], poses_sampled)
                with torch.no_grad(): #算回去xyr
                    xyz_pred, euler_pred = pose2euler(output["pred_pose"])
                    # output["rxyz_pred"] = torch.hstack([euler_pred[...,2].unsqueeze(1), xyz_pred])# 4DoF
                    output["rxyz_pred"] = torch.hstack([euler_pred, xyz_pred])
            
            del score_mean
            del score

            #计算方差，并根据方差确定下一阶段的范围
            samp_variance = (rxyz_sampled - output["rxyz_pred"].unsqueeze(1)) ** 2
            # output["exp_variance"] = lamb_ * (torch.sum(samp_variance * output["prob"].unsqueeze(2).expand(-1,-1,4), dim=1, keepdim=False) ** 0.5) # 4DoF

            output["exp_variance"] = lamb_ * (torch.sum(samp_variance * output["prob"].unsqueeze(2).expand(-1,-1, 6), dim=1, keepdim=False) ** 0.5)
            exp_var = output["exp_variance"]
            pred_pose = output["pred_pose"]
            # output["pred_pose"] = output["pred_pose"] @ transf.to(pred_pose)

            pred["stage{}".format(stage_idx + 1)] = output
        return {
            **pred,
            "pred_pose": output["pred_pose"],
            "rxyz_pred": output["rxyz_pred"],
        }
        

    def loss(self, pred, data):
        
        if self.loss_id == "softmax":
        ### SoftMax loss function
            nll = multi_stage_loss(pred, data['pose_GT'], self.loss_weight, self.num_stage)
        elif self.loss_id == "kl_loss":
        ### KL loss function
            nll = multi_stage_loss_KL(pred, data, self.loss_weight, self.num_stage)
        elif self.loss_id == "l1_loss":
        ### L1 loss function
            nll = multi_stage_loss_l1loss(pred, data['pose_GT'], self.loss_weight, self.num_stage)
        else:
            assert("Please input right loss function name.")

        loss = {"total": nll, "nll": nll}

        return loss

    def metrics(self): 
        return {
            "xyz_error": Location2DError("rxyz_pred"),
            "x_error": Location2DError_x("rxyz_pred"),
            "y_error": Location2DError_y("rxyz_pred"),
            "z_error": Location2DError_z("rxyz_pred"),
            "yaw_error": AngleError("rxyz_pred"),

            "xy_recall_2dot5m": Location2DRecall_xy(0.25,  "rxyz_pred"),
            "xy_recall_1m": Location2DRecall_xy(1.0,  "rxyz_pred"),
            "xy_recall_2m": Location2DRecall_xy (2.0,  "rxyz_pred"),
            "xy_recall_3m": Location2DRecall_xy(3.0,  "rxyz_pred"),
            "xy_recall_5m": Location2DRecall_xy(5.0,  "rxyz_pred"),
            "xy_recall_10m": Location2DRecall_xy(10.0,  "rxyz_pred"),
            "xy_recall_20m": Location2DRecall_xy(20.0,  "rxyz_pred"),

            "z_recall_2dot5m": Location2DRecall_z(0.25,  "rxyz_pred"),
            "z_recall_1m": Location2DRecall_z(1.0,  "rxyz_pred"),
            "z_recall_2m": Location2DRecall_z (2.0,  "rxyz_pred"),
            "z_recall_3m": Location2DRecall_z(3.0,  "rxyz_pred"),
            "z_recall_5m": Location2DRecall_z(5.0,  "rxyz_pred"),
            "z_recall_10m": Location2DRecall_z(10.0,  "rxyz_pred"),
            "z_recall_20m": Location2DRecall_z(20.0,  "rxyz_pred"),

            "xyz_recall_2dot5m": Location2DRecall(0.25,  "rxyz_pred"),
            "xyz_recall_1m": Location2DRecall(1.0,  "rxyz_pred"),
            "xyz_recall_2m": Location2DRecall (2.0,  "rxyz_pred"),
            "xyz_recall_3m": Location2DRecall(3.0,  "rxyz_pred"),
            "xyz_recall_5m": Location2DRecall(5.0,  "rxyz_pred"),
            "xyz_recall_10m": Location2DRecall(10.0,  "rxyz_pred"),
            "xyz_recall_20m": Location2DRecall(20.0,  "rxyz_pred"),

            "yaw_recall_1°": AngleRecall(1.0, "rxyz_pred"),
            "yaw_recall_2°": AngleRecall(2.0, "rxyz_pred"),
            "yaw_recall_3°": AngleRecall(3.0, "rxyz_pred"),
            "yaw_recall_5°": AngleRecall(5.0, "rxyz_pred"),
            "yaw_recall_7°": AngleRecall(7.0, "rxyz_pred"),
            "yaw_recall_10°": AngleRecall(10.0, "rxyz_pred"),
            
            "AllRecall_1m1°" : AllRecall(1.0, 1.0, "pred_pose"),
            "AllRecall_2m2°" : AllRecall(2.0, 2.0, "pred_pose"),
            "AllRecall_3m3°" : AllRecall(3.0, 3.0, "pred_pose"),
            "AllRecall_5m5°" : AllRecall(5.0, 5.0, "pred_pose"),
            "AllRecall_10m7°" : AllRecall(10.0, 7.0, "pred_pose"),
            "AllRecall_20m10°" : AllRecall(20.0, 10.0, "pred_pose")
        }
