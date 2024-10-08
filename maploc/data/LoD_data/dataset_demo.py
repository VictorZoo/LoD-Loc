# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

import json
from collections import defaultdict
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import DictConfig, OmegaConf

from ... import logger, DATASETS_PATH
from ..dataset import LoDLocDataset
from ..sequential import chunk_sequence
from ..torch import collate, worker_init_fn
from .utils import qvec2rotmat
from scipy.spatial.transform import Rotation as Rota

def pack_dump_dict(dump):
    for per_seq in dump.values():
        if "points" in per_seq:
            for chunk in list(per_seq["points"]):
                points = per_seq["points"].pop(chunk)
                if points is not None:
                    per_seq["points"][chunk] = np.array(
                        per_seq["points"][chunk], np.float64
                    )
        for view in per_seq["views"].values():
            for k in ["R_c2w", "roll_pitch_yaw"]:
                view[k] = np.array(view[k], np.float32)
            for k in ["chunk_id"]:
                if k in view:
                    view.pop(k)
        if "observations" in view:
            view["observations"] = np.array(view["observations"])
        for camera in per_seq["cameras"].values():
            for k in ["params"]:
                camera[k] = np.array(camera[k], np.float32)
    return dump

def trans_euler(q_vec):
    q, t = q_vec[:4], q_vec[4:]

    R = np.asmatrix(qvec2rotmat(q))   
    t = -R.T @ t
    R = R.T

    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3,3] = t   #!  c2w

    transf = np.array([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1.],
    ])
    T = T @ transf

    R_gt = T[:3, :3]
    ret_gt = Rota.from_matrix(R_gt)
    euler_gt = ret_gt.as_euler('xyz',degrees=True)
    t_gt = list(T[:3, 3])
    euler_gt = list(euler_gt)

    return euler_gt+t_gt



class DEMODataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    images_archive = "images.tar.gz"
    images_dirname = "images/"
    query_dirname = "Query_image/"
    mask_dirname = "Line_mask/"

    default_cfg = {
        **LoDLocDataset.default_cfg,
        "name": "DEMO",
        # paths and fetch
        "data_dir": DATASETS_PATH / "DEMO", #改，data dir
        "local_dir": None,
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0, "num_sample": 2000, "if_save": False, 'interval': 1},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        self.interval = self.cfg.loading.val.interval
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL")

        # if self.cfg.crop_size_meters < self.cfg.max_init_error:
        #     raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        pass
        

    def parse_pose_list(self, path, scene):
        with open(path, 'r') as f:
            for data in f.read().rstrip().split('\n'):
                data = data.split()
                name = data[0].split('/')[-1]
                q = np.array(data[1:], float)
                euler_GT = trans_euler(q)
                self.names_setup.append((scene, name, euler_GT))

    def setup(self, stage: Optional[str] = None):
        self.image_dirs = {}
        self.GTpose_dirs = {}
        self.GPSpose_dirs = {}
        self.intrin_dirs = {}
        self.points3D_dirs = {}
        self.mask_dirs = {}
        self.names_setup = []
        
        for scene in self.cfg.scenes:

            logger.info("Loading scene %s.", scene)

            pose_pth = os.path.join(self.root, scene, 'GT_pose.txt')
            # GPSpose_pth = os.path.join(self.root, scene, 'GT_pose.txt') # no GPS for Synthesis
            self.parse_pose_list(pose_pth, scene)
            self.image_dirs[scene] = (
                (self.local_dir or self.root) / scene / 'Query_image' 
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            self.GTpose_dirs[scene] =(
                (self.local_dir or self.root) / scene / 'GT_pose' 
            )
            assert self.GTpose_dirs[scene].exists(), self.GTpose_dirs[scene]

            self.GPSpose_dirs[scene] =(
                (self.local_dir or self.root) / scene / 'GPS_pose' 
            )
            assert self.GPSpose_dirs[scene].exists(), self.GPSpose_dirs[scene]


            self.intrin_dirs[scene] =(
                (self.local_dir or self.root) / scene / 'intrinsic' 
            )
            assert self.intrin_dirs[scene].exists(), self.intrin_dirs[scene]

            self.points3D_dirs[scene] =(
                (self.local_dir or self.root) / scene / 'Points' 
            )
            assert self.points3D_dirs[scene].exists(), self.points3D_dirs[scene]

            self.mask_dirs[scene] =(
                (self.local_dir or self.root) / scene / 'Mask' 
            )
            assert self.points3D_dirs[scene].exists(), self.points3D_dirs[scene]
        # names_setup = [scene, name, euler_GT, euler_GPS]
        self.parse_splits(self.cfg.split, self.names_setup)



    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, float):
            split_arg = int(len(names) * split_arg)
            
            # names = np.random.RandomState(self.cfg.seed).permutation(names)#.tolist()
            np.random.seed(self.cfg.seed)          
            random.shuffle(names)
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
            
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }

        elif isinstance(split_arg, str):
            with (self.root / split_arg).open("r") as fp:
                splits = json.load(fp)
            splits = {
                k: {loc: set(ids) for loc, ids in split.items()}
                for k, split in splits.items()
            }
            self.splits = {}
            for k, split in splits.items():
                self.splits[k] = [
                    n
                    for n in names
                    if n[0] in split and n[1] in split[n[0]]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return LoDLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.splits[stage],
            self.image_dirs,
            self.GTpose_dirs,
            self.GPSpose_dirs,
            self.intrin_dirs,
            self.points3D_dirs,
            self.mask_dirs,
            self.interval,
            image_ext=".JPG",
            points3D_ext=".npy",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ): 
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx
