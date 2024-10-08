# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf

from ..utils.io import read_image
from .utils import parse_pose_list, point_proj, read_intrinsic


class LoDLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # image preprocessing
        "resize_image": None,
        "augmentation": {
            "image": {
                "apply": False,
                "brightness": [0.8, 1.6],
                "contrast": [0.8, 1.2]
            },
        },
    }
 
    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        GTpose_dirs: Dict[str, Path],
        GPSpose_dirs:Dict[str, Path],
        intrin_dirs: Dict[str, Path],
        points3D_dirs: Dict[str, Path],
        interval: int,
        image_ext: str = "",
        points3D_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.GTpose_dirs = GTpose_dirs
        self.GPSpose_dirs = GPSpose_dirs
        self.intrin_dirs = intrin_dirs
        self.points3D_dirs = points3D_dirs
        self.names = names
        self.image_ext = image_ext
        self.points3D_ext = points3D_ext
        self.interval = str(interval)
        self.transform = transforms.Compose([transforms.Resize(self.cfg.resize_image), transforms.ToTensor()])

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            # args = OmegaConf.masked_copy(
            #     cfg.augmentation.image, ["brightness", "contrast"]
            # )
            # tfs.append(tvf.ColorJitter(**args))
            tfs.append(tvf.ColorJitter(brightness=list(cfg.augmentation.image.brightness), \
                                        contrast=list(cfg.augmentation.image.contrast)))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        scene, name, pose_GT = self.names[idx] 
        data_i = {
            "index": idx,
            "name": name,
            "scene": scene,
            "pose_GT": torch.tensor(pose_GT), #再加一个IMAGE,tanspose之后的
        }
        
        image = read_image(self.image_dirs[scene] / (name)) # + self.image_ext)) .split('.')[0]+'.png'
        h_origin, w_origin,_ = image.shape
        data_i['origin_hw'] = torch.tensor([h_origin, w_origin])
        
        image = self.process_image(image, self.cfg.resize_image, seed)
        data_i['image'] = image

        # 可以先把投影二维点存起来!
        # points3D = np.load(self.points3D_dirs[scene] / (name.split(".")[0].split("_img")[0]+ '_points' + self.points3D_ext))
        # np.random.seed(seed)
        # obs = np.random.choice(len(points3D), self.cfg.loading.train.num_sample)
        # points3D = points3D[obs]
        # data_i['points3D'] = torch.from_numpy(points3D)

        # 测试的时候用
        if self.stage == "train":
            points3D = np.load(self.points3D_dirs[scene] / ('0.5m') / (name.split(".")[0].split("_img")[0]+ '_points' + self.points3D_ext))
            np.random.seed(seed)
            obs = np.random.choice(len(points3D), self.cfg.loading.train.num_sample)
            points3D = points3D[obs]
            data_i['points3D'] = torch.from_numpy(points3D)
        else:
            points3D = np.load(self.points3D_dirs[scene] / (self.interval+'m') /(name.split(".")[0].split("_img")[0]+ '_points' + self.points3D_ext))
            if len(points3D) >=20000:
                np.random.seed(seed)
                obs = np.random.choice(len(points3D), 20000)
                points3D = points3D[obs]
                data_i['points3D'] = torch.from_numpy(points3D)
            else:
                data_i['points3D'] = torch.from_numpy(points3D)

        data_i['pose_sample'] = np.loadtxt(self.GPSpose_dirs[scene] / (name.split(".")[0].split("_img")[0] + "_pose.txt")) #C2Wqq
        data_i['pose_GT_4x4'] = np.loadtxt(self.GTpose_dirs[scene] / (name.split(".")[0].split("_img")[0] + "_pose.txt")) #C2W
        data_i['intrinsic'] = np.loadtxt(self.intrin_dirs[scene] / (name.split(".")[0].split("_img")[0] + "_intrinsic.txt")) # W2C

        return data_i 

    def process_image(self, image, resize_, seed):
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        h_new, w_new = resize_
        image = tvf.functional.resize(image, (h_new, w_new), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)

        return image

    def proj_pose(self, sample_pth, k_pth, points3D):
        origin_corr = torch.tensor([0, 0, 0]) # 改超参数
        pose_sample = torch.from_numpy(parse_pose_list(sample_pth))
        k_, w, h = read_intrinsic(k_pth)
        final_sampled_uv, _ = point_proj(points3D, pose_sample.cuda(), torch.from_numpy(k_).cuda(), origin_corr, w, h, self.cfg.loading.train.num_sample)

        return final_sampled_uv.cpu()
    

class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # image preprocessing
        "resize_image": None,
        "augmentation": {
            "image": {
                "apply": False,
                "brightness": [0.8, 1.6],
                "contrast": [0.8, 1.2]
            },
        },
    }
 
    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        GTpose_dirs: Dict[str, Path],
        GPSpose_dirs:Dict[str, Path],
        intrin_dirs: Dict[str, Path],
        points3D_dirs: Dict[str, Path],
        mask_dirs: None,
        interval: int,
        image_ext: str = "",
        points3D_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.GTpose_dirs = GTpose_dirs
        self.GPSpose_dirs = GPSpose_dirs
        self.intrin_dirs = intrin_dirs
        self.points3D_dirs = points3D_dirs
        self.mask_dirs = mask_dirs
        self.names = names
        self.image_ext = image_ext
        self.points3D_ext = points3D_ext
        self.interval = str(interval)
        self.transform = transforms.Compose([transforms.Resize(self.cfg.resize_image), transforms.ToTensor()])

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            # args = OmegaConf.masked_copy(
            #     cfg.augmentation.image, ["brightness", "contrast"]
            # )
            # tfs.append(tvf.ColorJitter(**args))
            tfs.append(tvf.ColorJitter(brightness=list(cfg.augmentation.image.brightness), \
                                        contrast=list(cfg.augmentation.image.contrast)))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        scene, name, pose_GT = self.names[idx] 
        data_i = {
            "index": idx,
            "name": name,
            "scene": scene,
            "pose_GT": torch.tensor(pose_GT), #再加一个IMAGE,tanspose之后的
        }
        
        image = read_image(self.image_dirs[scene] / (name)) # + self.image_ext)) .split('.')[0]+'.png'
        h_origin, w_origin,_ = image.shape
        data_i['origin_hw'] = torch.tensor([h_origin, w_origin])
        
        image = self.process_image(image, self.cfg.resize_image, seed)
        data_i['image'] = image

        # 可以先把投影二维点存起来!
        # points3D = np.load(self.points3D_dirs[scene] / (name.split(".")[0].split("_img")[0]+ '_points' + self.points3D_ext))
        # np.random.seed(seed)
        # obs = np.random.choice(len(points3D), self.cfg.loading.train.num_sample)
        # points3D = points3D[obs]
        # data_i['points3D'] = torch.from_numpy(points3D)

        # 测试的时候用
        if self.stage == "train":
            points3D = np.load(self.points3D_dirs[scene] / ('0.5m') / (name.split(".")[0].split("_img")[0]+ '_points' + self.points3D_ext))
            np.random.seed(seed)
            obs = np.random.choice(len(points3D), self.cfg.loading.train.num_sample)
            points3D = points3D[obs]
            data_i['points3D'] = torch.from_numpy(points3D)
        else:
            points3D = np.load(self.points3D_dirs[scene] / (self.interval+'m') /(name.split(".")[0].split("_img")[0]+ '_points' + self.points3D_ext))
            if len(points3D) >=20000:
                np.random.seed(seed)
                obs = np.random.choice(len(points3D), 20000)
                points3D = points3D[obs]
                data_i['points3D'] = torch.from_numpy(points3D)
            else:
                data_i['points3D'] = torch.from_numpy(points3D)

        data_i['pose_sample'] = np.loadtxt(self.GPSpose_dirs[scene] / (name.split(".")[0].split("_img")[0] + "_pose.txt")) #C2Wqq
        data_i['pose_GT_4x4'] = np.loadtxt(self.GTpose_dirs[scene] / (name.split(".")[0].split("_img")[0] + "_pose.txt")) #C2W
        data_i['intrinsic'] = np.loadtxt(self.intrin_dirs[scene] / (name.split(".")[0].split("_img")[0] + "_intrinsic.txt")) # W2C

        return data_i 

    def process_image(self, image, resize_, seed):
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        h_new, w_new = resize_
        image = tvf.functional.resize(image, (h_new, w_new), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)

        return image

    def proj_pose(self, sample_pth, k_pth, points3D):
        origin_corr = torch.tensor([0, 0, 0]) # 改超参数
        pose_sample = torch.from_numpy(parse_pose_list(sample_pth))
        k_, w, h = read_intrinsic(k_pth)
        final_sampled_uv, _ = point_proj(points3D, pose_sample.cuda(), torch.from_numpy(k_).cuda(), origin_corr, w, h, self.cfg.loading.train.num_sample)

        return final_sampled_uv.cpu()
