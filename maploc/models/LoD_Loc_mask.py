# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.
import torch
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
from . import get_model
from .base import BaseModel
from .submodules import *
import os
from .utils import(
    point_proj, 
    sample_poses, 
    find_max, 
    pose2euler,
    euler2pose,
    multi_stage_loss, 
    multi_stage_loss_l1loss,
    norm_uv, 
    visualize_feature_map,
    multi_stage_loss_KL,trans_W2C,
    transf)
from .voting import (
    log_softmax_spatial,
    softmax_spatial,
    get_score,
    get_mean_score,
)
from .metrics import (AngleError, AngleRecall, Location2DError_x, Location2DError_y,Location2DError_z,Location2DError, 
                        Location2DRecall_xy, Location2DRecall_z,Location2DRecall,AllRecall ,AllRecall_6DoF, RError)
from ..utils.geometry.wrappers import Pose, Camera


class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)

class LoD_Loc(BaseModel):
    default_conf = {
        "num_sample": "???",
        "num_sample_val": "???",
        "error_ranges" : "???",
        "lamb": "???",
        "base_chs": [8,8,8],
        "grad_method": "Detach",
        "stage_configs": "???",
        "lamb" : "???",
        "lamb_val" : "???",
        "feat_ext_ch" : "???",
        "loss_weight" : "???",
        "confidence" : "???",
        "thresh": 0.5,
        "loss_id": "l1_loss",
        "save_pth": None,
        "refine": True, 
    }

    def _init(self, conf):
        self.stage_configs = self.conf.stage_configs
        self.lamb = self.conf.lamb
        self.lamb_val = self.conf.lamb_val
        self.error_ranges = self.conf.error_ranges
        self.num_sample = self.conf.num_sample
        self.num_sample_val = self.conf.num_sample_val
        self.num_stage = len(self.conf.stage_configs)
        self.loss_weight = self.conf.loss_weight
        self.loss_id = self.conf.loss_id
        self.confidence = self.conf.confidence
        self.save_pth = self.conf.save_pth
        self.refine = conf.refine

        self.feature_extraction = FeatExtNet(base_channels=self.conf.feat_ext_ch, num_stage=3)

    def interpolate_feature_map(self, feature, p2d, return_gradients=False):
        interpolation_pad = 4
        b, c, h, w = feature.shape
        scale = torch.tensor([w-1, h-1]).to(p2d)
        pts = (p2d / scale) * 2  - 1
        pts = pts.clamp(min=-2, max=2)
        fp = torch.nn.functional.grid_sample(feature, pts[:, None], mode='bilinear', align_corners=True)
        fp = fp.reshape(b, c, -1).transpose(-1, -2)
        
        image_size_ = torch.tensor([w-interpolation_pad-1, h-interpolation_pad-1]).to(pts)
        valid = torch.all((p2d >= interpolation_pad) & (p2d <= image_size_), -1)

        if return_gradients:
            dxdy = torch.tensor([[1, 0], [0, 1]])[:, None].to(pts) / scale * 2
            dx, dy = dxdy.chunk(2, dim=0)
            pts_d = torch.cat([pts-dx, pts+dx, pts-dy, pts+dy], 1)
            tensor_d = torch.nn.functional.grid_sample(
                    feature, pts_d[:, None], mode='bilinear', align_corners=True)
            tensor_d = tensor_d.reshape(b, c, -1).transpose(-1, -2)
            tensor_x0, tensor_x1, tensor_y0, tensor_y1 = tensor_d.chunk(4, dim=1)
            gradients = torch.stack([
                (tensor_x1 - tensor_x0)/2, (tensor_y1 - tensor_y0)/2], dim=-1)
        else:
            gradients = torch.zeros(b, pts.shape[1], c, 2).to(feature)

        return fp, valid, gradients
    
    def run_refine(self, vertices, pose, fmap, cam, image=None):
        p2d, visible = cam.view2image(pose.transform(vertices))
        fp, valid, J_f = self.interpolate_feature_map(fmap, p2d, return_gradients=True)

        valid = (valid & visible).detach()
        
        res = -fp # 1 - fp
        J_p3d_pose = pose.R[:, None] @ pose.J_transform(vertices)
        J_p2d_p3d, _ = cam.J_world2image(pose.transform(vertices))
        J = -J_f @ J_p2d_p3d @ J_p3d_pose
        
        weight = valid.float()
        
        grad = torch.einsum('...ndi,...nd->...ni', J, res) 
        grad = weight[..., None] * grad
        grad = grad.sum((-2))
        
        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
        Hess = weight[..., None, None] * Hess
        Hess = Hess.sum((-3))
        
        if image is not None:
            import cv2
            tmp_image = image.mul(255).byte().permute(0, 2, 3, 1).cpu().detach().numpy().copy()
            for i, display_image in enumerate(tmp_image):
                d_p2d = p2d[i].cpu().detach().numpy()
                for p in d_p2d:
                    cv2.circle(display_image, (int(p[0]), int(p[1])), 1, (255, 0, 0), 1)                  
                cv2.imwrite(f'test-{i}.png', display_image)
        
        return -grad.unsqueeze(-1), Hess
    
    def _forward(self, data):
        pred = {}
        features = self.feature_extraction(data["image"])
        exp_var, pred_pose = None, None
        lamb_ = None
        
        # 根据不同stage进行操作
        for stage_idx in range(self.num_stage): #self.num_stage
            output = {}
            
            if data['epoch_stage'] == 'train':
                lamb_ = self.lamb[stage_idx]
            elif data['epoch_stage'] == 'val':
                lamb_ = self.lamb_val[stage_idx]
            features_stage = features["stage{}_f".format(stage_idx + 1)]
            conf_feature = features["stage{}_c".format(stage_idx + 1)]

            if self.confidence:
                f_weight = features_stage  * conf_feature
            else:  
                f_weight = features_stage
            
            if self.save_pth:
                if not os.path.exists(self.save_pth):
                    os.mkdir(self.save_pth)
                
                save_img_pth = os.path.join(self.save_pth, data['name'][0].split('.')[0]+'_'+str(stage_idx)+'.pth')
                torch.save(f_weight.cpu().squeeze(), save_img_pth)
                # visualize_feature_map(f_weight.cpu().squeeze(), save_img_pth)
                save_refine_pth = os.path.join(self.save_pth, data['name'][0].split('.')[0]+'_refine'+'.pth')
                torch.save(features['stage_fine'].cpu().squeeze(), save_refine_pth)
                # visualize_feature_map(-features['stage_fine'].cpu().squeeze(), save_refine_pth)
                
            
            if stage_idx == 0:
                # 初始采样范围
                ranges = torch.tensor(self.error_ranges)
                output["ranges"] = ranges
               
                if data['epoch_stage'] == 'train':
                    output["num_sample"] = torch.tensor(self.num_sample[stage_idx])
                elif data['epoch_stage'] == 'val':
                    output["num_sample"] = torch.tensor(self.num_sample_val[stage_idx])

                poses_init = data['pose_sample'] @ transf.to(data['pose_sample'])
            # else:
            #     # 不用方差
            #     ranges = torch.tensor(self.error_ranges)
            #     output["ranges"] = ranges
            #     output["num_sample"] = torch.tensor(self.num_sample[stage_idx])
            #     poses_init = data['pose_sample'] @ transf.to(data['pose_sample'])
            else:
                # 根据上一阶段计算的方差来确定range范围
                low_bound = -exp_var
                high_bound = exp_var
                ranges = torch.stack((low_bound, high_bound),dim = 1).permute(0, 2 ,1)
                
                if data['epoch_stage'] == 'train':
                    output["num_sample"] = torch.tensor(self.num_sample[stage_idx])
                elif data['epoch_stage'] == 'val':
                    output["num_sample"] = torch.tensor(self.num_sample_val[stage_idx])

                output["ranges"] = ranges
                poses_init = pred_pose @ transf.to(pred_pose)
            
            # Pose采样、投影并取Feature
            poses_sampled, rxyz_sampled, output["sample_euler"], PitchRoll = sample_poses(poses_init[:,0:3,3], poses_init[:,0:3,0:3], ranges, output["num_sample"], data['pose_GT'].squeeze())
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
                    bind_eulerPose = torch.cat([PitchRoll, output["rxyz_pred"]],dim=1)
                    output["pred_pose"] = euler2pose(bind_eulerPose[:,:3], bind_eulerPose[:,3:])
                
            ### Softmax Loss 时需要用的步骤
            elif self.loss_id == "softmax" or "kl_loss":
                output["log_prob"] = log_softmax_spatial(score_mean, dims = 1)
                output["w_feature"] = f_weight
                output["pred_score_mean"], output["pred_pose"], output["pred_score"] = find_max(output["log_prob"], poses_sampled, score)
                # output["pred_score_mean"], output["pred_pose"] = find_max(output["prob"], poses_sampled)
                with torch.no_grad(): #算回去xyr
                    xyz_pred, euler_pred = pose2euler(output["pred_pose"])
                    output["rxyz_pred"] = torch.hstack([euler_pred[...,2].unsqueeze(1), xyz_pred])
            
            del score_mean
            del score

            #计算方差，并根据方差确定下一阶段的范围
            samp_variance = (rxyz_sampled - output["rxyz_pred"].unsqueeze(1)) ** 2
            output["exp_variance"] = lamb_ * (torch.sum(samp_variance * output["prob"].unsqueeze(2).expand(-1,-1,4), dim=1, keepdim=False) ** 0.5)
            
            exp_var = output["exp_variance"]
            pred_pose = output["pred_pose"]

            pred["stage{}".format(stage_idx + 1)] = output
            pred_pose_, rxyz_pred_ = output["pred_pose"], output["rxyz_pred"]
        
        if self.refine:
            camera_pose = Pose.from_4x4mat(pred_pose)
            poses_init_ = camera_pose.inv()
            
            feature_map = features['stage_fine']
            _, _, new_h, new_w = feature_map.shape
            scale_factors = torch.stack([(new_h-1) / data['origin_hw'][:, 0], (new_w-1) / data['origin_hw'][:, 1]], dim=1)
            
            intrinsic = data['intrinsic']
            fx = intrinsic[:, 0, 0] * scale_factors[:, 1]
            fy = intrinsic[:, 1, 1] * scale_factors[:, 0]
            cx = intrinsic[:, 0, 2] * scale_factors[:, 1]
            cy = intrinsic[:, 1, 2] * scale_factors[:, 0]
            hw = data['origin_hw']
            height = hw[:, 0] * scale_factors[:, 0]
            width = hw[:, 1] * scale_factors[:, 1]
            camera_intrics = torch.stack([width, height, fx, fy, cx, cy], dim=-1).float()
            camera = Camera(camera_intrics)
            
            for it in range(5):
                grad, Hess = self.run_refine(data["points3D"], poses_init_, feature_map, camera) # , data['image'])
                
                A = Hess.cpu()
                B = grad.cpu()
                diag = A.diagonal(dim1=-2, dim2=-1) * 0.01 + 0.0001
                A = A + diag.diag_embed()
                try:
                    U = torch.linalg.cholesky(A)
                except RuntimeError as e:
                    if 'singular U' in str(e):
                        print('Cholesky decomposition failed, fallback to LU.')
                        try:
                            delta = torch.linalg.solve(A, B)[..., 0]
                        except RuntimeError:
                            delta = torch.zeros_like(B)[..., 0]
                            print('A is not invertible')
                    else:
                        raise
                else:
                    delta = torch.cholesky_solve(B, U)[..., 0]
                
                delta = delta.to(poses_init_.device)
                aa, t = delta.split([3, 3], dim=-1)
                delta_pose = Pose.from_aa(aa, t)
                poses_init_ = poses_init_ @ delta_pose
            
            gt_pose = data['pose_GT_4x4'].float()
            gt_pose = Pose.from_4x4mat(gt_pose).inv()
            output_refine = {
                "gt_poses_refine": gt_pose,
                "pred_poses_refine": poses_init_,
                "camera_refine": camera,
            }
        
            pred["stage_refine"] = output_refine

            # 计算4x4的pose和euler
            pred_R, pred_t = poses_init_.R, poses_init_.t
            with torch.no_grad():
                pred_pose_, rxyz_pred_ = trans_W2C(pred_R, pred_t)

            
        return {
            **pred,
            "rxyz_pred": rxyz_pred_,
            "pred_pose": pred_pose_,
            "fine_feat":features['stage_fine'],
        }
        

    def loss(self, pred, data):
        
        if self.loss_id == "softmax":
        ### SoftMax loss function
            nll = multi_stage_loss(pred, data['pose_GT'][...,2:], self.loss_weight, self.num_stage)
        elif self.loss_id == "kl_loss":
        ### KL loss function
            nll = multi_stage_loss_KL(pred, data, self.loss_weight, self.num_stage)
        elif self.loss_id == "l1_loss":
        ### L1 loss function
            nll = multi_stage_loss_l1loss(pred, data['pose_GT'][...,2:], self.loss_weight, self.num_stage)
        else:
            assert("Please input right loss function name.")
        
        loss = {"total": nll, "nll": nll.clone()}
        
        if self.refine: # and data['current_epoch'] > -10:
            from ..utils.geometry.losses import scaled_barron
            
            pred_refine = pred['stage_refine']
            gt_poses_refine = pred_refine['gt_poses_refine']
            pred_poses_refine = pred_refine['pred_poses_refine']
            camera_refine = pred_refine['camera_refine']
            # p3d = data['points3D']
            
            def project(pose, p3d):
                p3d_in_cam = pose.transform(p3d)
                return camera_refine.view2image(p3d_in_cam)

            def masked_mean(x, mask, dim, confindence=None):
                mask = mask.float()
                if confindence is not None:
                    mask *= confindence
                return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)
            
            def reprojection_error(pose, p3d, gt, valid):
                p2d, _ = project(pose, p3d)
                err = torch.sum((gt - p2d) ** 2, dim=-1)
                err = scaled_barron(1., 2.)(err)[0] / 4
                err = masked_mean(err, valid, -1)
                return err
            
            gt_vertex_in_image, gt_vertex_valid = project(gt_poses_refine, data['points3D'])
            err_reprojection = reprojection_error(pred_poses_refine, data['points3D'], gt_vertex_in_image,
                                                  gt_vertex_valid) #.clamp(max=50)
            err_reprojection = err_reprojection * 0.25 # 0.25
            err_reprojection = err_reprojection.clamp(max=5)

            loss['"refine_reprojection_err"'] = err_reprojection
            loss['total'] += err_reprojection

        return loss

    def metrics(self): 
        return {
            "xyz_error": Location2DError("rxyz_pred"),
            "x_error": Location2DError_x("rxyz_pred"),
            "y_error": Location2DError_y("rxyz_pred"),
            "z_error": Location2DError_z("rxyz_pred"),
            "yaw_error": AngleError("rxyz_pred"),
            "R_error": RError("pred_pose"),

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

            "AllRecall6DoF_1m1°" : AllRecall_6DoF(1.0, 1.0, "pred_pose"),
            "AllRecall6DoF_2m2°" : AllRecall_6DoF(2.0, 2.0, "pred_pose"),
            "AllRecall6DoF_3m3°" : AllRecall_6DoF(3.0, 3.0, "pred_pose"),
            "AllRecall6DoF_5m5°" : AllRecall_6DoF(5.0, 5.0, "pred_pose"),
            "AllRecall6DoF_10m7°" : AllRecall_6DoF(10.0, 7.0, "pred_pose"),
            "AllRecall6DoF_20m10°" : AllRecall_6DoF(20.0, 10.0, "pred_pose")
        }
