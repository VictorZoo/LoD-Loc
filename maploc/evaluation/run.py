# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

import functools
from itertools import islice
import os
from typing import Callable, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MetricCollection
from pytorch_lightning import seed_everything
from tqdm import tqdm

from .. import logger, EXPERIMENTS_PATH
from ..data.torch import collate, unbatch_to_device
from ..models.metrics import AngleError, LateralLongitudinalError, Location2DError
from ..module import GenericModule
from ..utils.io import download_file, DATA_URL
from .utils import write_dump


pretrained_models = dict(
    JadeBay_ckpt=("JadeBay.ckpt", dict(num_rotations=256)),
)

exp_path = os.path.join(Path(__file__).parent.parent.parent,"experiments")

def save_results(GPS_folder, GT_folder):
    result = GPS_folder + ".txt"
    list = os.listdir(GPS_folder)
    image_num = len(list)
    errors_t = []
    errors_R = []
    for item in list:
        
        GPS_pth = os.path.join(GPS_folder, item)
        GT_pth = os.path.join(GT_folder, item)

        GPS_pose = np.loadtxt(GPS_pth)
        GT_pose = np.loadtxt(GT_pth)

        e_t = np.linalg.norm(GPS_pose[:3,3] - GT_pose[:3,3], axis=0)

        cos = np.clip((np.trace(np.dot(GPS_pose[:3,:3], GT_pose[:3,:3].T)) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))

        errors_t.append(e_t)
        errors_R.append(e_R)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    max_t = np.max(errors_t)
    max_R = np.max(errors_R)
    min_t = np.min(errors_t)
    min_R = np.min(errors_R)

    out = f'\nTest image nums: {image_num}'
    out += f'\nMedian errors: {med_t*100:.3f}cm, {med_R:.3f}deg'
    out += f'\nMax errors: {max_t*100:.3f}cm, {max_R:.3f}deg'
    out += f'\nMin errors: {min_t*100:.3f}cm, {min_R:.3f}deg'
    out += '\nPercentage of test images localized within:'

    threshs_t = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 7.0, 7.0, 10.0, 30.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio_all = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f'\n\t{th_t*1:.0f}m, {th_R:.0f}deg : {ratio_all*100:.2f}%'

    print(out)
    with open(result,'w') as f:
        f.writelines(out)

def resolve_checkpoint_path(experiment_or_path: str) -> Path:
    path = Path(experiment_or_path)
    if not path.exists():
        # provided name of experiment
        path = Path(EXPERIMENTS_PATH, *experiment_or_path.split("/"))
        if not path.exists():
            if experiment_or_path in set(p for p, _ in pretrained_models.values()):
                download_file(f"{DATA_URL}/{experiment_or_path}", path)
            else:
                raise FileNotFoundError(path)
    if path.is_file():
        return path
    # provided only the experiment name
    maybe_path = path / "last-step.ckpt"
    if not maybe_path.exists():
        maybe_path = path / "step.ckpt"
    if not maybe_path.exists():
        raise FileNotFoundError(f"Could not find any checkpoint in {path}.")
    return maybe_path


@torch.no_grad()
def evaluate_single_image(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
):

    metrics = MetricCollection(model.model.metrics())

    metrics = metrics.to(model.device)

    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)
        batch['epoch_stage'] = 'val'
        pred = model(batch)
        
        results = metrics(pred, batch)
        if callback is not None:
            callback(
                i, model, unbatch_to_device(pred), unbatch_to_device(batch_), results
            )
        del batch_, batch, pred, results

    return metrics.cpu()

@torch.no_grad()
def evaluate_single_image_multistage(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    stage: int,
    save: Path,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
):
    metrics = {}
    for i in range(stage):
        metrics['stage{}'.format(i+1)] = MetricCollection(model.model.metrics())

        metrics['stage{}'.format(i+1)] = metrics['stage{}'.format(i+1)].to(model.device)

    metrics['refine'] = MetricCollection(model.model.metrics())
    metrics['refine'] = metrics['refine'].to(model.device)

    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)
        batch['epoch_stage'] = 'val'
        pred = model(batch)
        for j in range(len(batch['name'])):
            
            # Final_pred
            save_pth = Path(os.path.join(save,"refine"))
            save_pth.mkdir(exist_ok=True, parents=True)
            save_pth = os.path.join(save_pth, batch['name'][j].split('.')[0]+'_pose.txt')
            wirt = np.array(pred['pred_pose'][j].cpu())
            np.savetxt(save_pth, wirt)
            for i in range(stage):
                
                save_pth = Path(os.path.join(save,'{}'.format(i+1)))
                save_pth.mkdir(exist_ok=True, parents=True)
                save_pth = os.path.join(save_pth, batch['name'][j].split('.')[0]+'_pose.txt')
                wirt = np.array(pred["stage{}".format(i+1)]['pred_pose'].squeeze().cpu())
                np.savetxt(save_pth, wirt)
          
        for i in range(stage):
            results = metrics['stage{}'.format(i+1)](pred["stage{}".format(i+1)], batch)
            if callback is not None:
                callback(
                    i, model, unbatch_to_device(pred), unbatch_to_device(batch_), results
                )

        results = metrics['refine'](pred, batch)
        if callback is not None:
            callback(
                i, model, unbatch_to_device(pred), unbatch_to_device(batch_), results
            )

        del pred, batch, batch_, results


    return metrics

@torch.no_grad()
def evaluate_single_image_demo(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    stage: int,
    save: Path,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,  
):
    

    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)
        batch['epoch_stage'] = 'val'
        pred = model(batch)
        for j in range(len(batch['name'])):
            
            # Final_pred
            save_pth = Path(os.path.join(save,"refine"))
            save_pth.mkdir(exist_ok=True, parents=True)
            save_pth = os.path.join(save_pth, batch['name'][j].split('.')[0]+'_pose.txt')
            wirt = np.array(pred['pred_pose'][j].cpu())
            np.savetxt(save_pth, wirt)
            feat_ = pred['fine_feat']
            torch.save(feat_.cpu().squeeze(), save_pth.split('_pose')[0]+'.pth')
            for i in range(stage):
                
                save_pth = Path(os.path.join(save,'{}'.format(i+1)))
                save_pth.mkdir(exist_ok=True, parents=True)
                save_pth = os.path.join(save_pth, batch['name'][j].split('.')[0]+'_pose.txt')
                wirt = np.array(pred["stage{}".format(i+1)]['pred_pose'].squeeze().cpu())
                np.savetxt(save_pth, wirt)
                feat_ = pred["stage{}".format(i+1)]['w_feature']
                torch.save(feat_.cpu().squeeze(), save_pth.split('_pose')[0]+'.pth')
        print("Save prediction to", save)

        del pred, batch, batch_
    

def evaluate(
    experiment: str,
    cfg: DictConfig,
    dataset,
    split: str,
    multstage: int,
    output_name: Optional[str] = None,
    callback: Optional[Callable] = None,
    num_workers: int = 1,
    viz_kwargs=None,
    **kwargs,
):
    if experiment in pretrained_models:
        experiment, cfg_override = pretrained_models[experiment]
        cfg = OmegaConf.merge(OmegaConf.create(dict(model=cfg_override)), cfg)

    logger.info("Evaluating model %s with config %s ", experiment, cfg)
    checkpoint_path = resolve_checkpoint_path(experiment)
    model = GenericModule.load_from_checkpoint(
        checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt")
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset.prepare_data()
    dataset.setup()

    if output_name is not None:
        if not experiment.endswith(".ckpt") :
            output_dir = Path(os.path.join(exp_path, experiment, output_name))
        else:
            output_dir = Path(os.path.join(exp_path, experiment.split('/')[0], output_name))
        output_dir.mkdir(exist_ok=True, parents=True)

    kwargs = {**kwargs, "callback": callback}
    save_pth = Path(os.path.join(output_dir,"pred_pose"))
    save_pth.mkdir(exist_ok=True, parents=True)
    seed_everything(dataset.cfg.seed)

    if multstage:
        loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)
        metrics = evaluate_single_image_multistage(loader, model, multstage, save_pth, **kwargs)
        for i in range(multstage):
            metrics_ = metrics['stage{}'.format(i+1)].cpu()
            results = metrics_.compute()
            logger.info("*** Below is stage%s's results! ***", i+1)
            logger.info("All results: %s", results)
            if output_dir is not None:
                write_dump(output_dir, experiment, cfg, results, i+1)
                logger.info("Outputs have been written to %s.", output_dir)

        metrics_ = metrics['refine'].cpu()
        results = metrics_.compute()
        logger.info("*** Below is stage%s's results! ***", "refine")
        logger.info("All results: %s", results)
        if output_dir is not None:
            write_dump(output_dir, experiment, cfg, results, "refine")
            logger.info("Outputs have been written to %s.", output_dir)
                
    else:
        loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)
        metrics = evaluate_single_image(loader, model, **kwargs)
        results = metrics.compute()
        logger.info("All results: %s", results)
        if output_dir is not None:
            write_dump(output_dir, experiment, cfg, results, "refine")
            logger.info("Outputs have been written to %s.", output_dir)
    
    return metrics


def evaluate_demo(
    experiment: str,
    cfg: DictConfig,
    dataset,
    split: str,
    multstage: int,
    output_name: Optional[str] = None,
    callback: Optional[Callable] = None,
    num_workers: int = 1,
    **kwargs,
):
    if experiment in pretrained_models:
        experiment, cfg_override = pretrained_models[experiment]
        cfg = OmegaConf.merge(OmegaConf.create(dict(model=cfg_override)), cfg)

    logger.info("Evaluating model %s", experiment)
    checkpoint_path = resolve_checkpoint_path(experiment)
    model = GenericModule.load_from_checkpoint(
        checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt")
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset.prepare_data()
    dataset.setup()

    if output_name is not None:
        if not experiment.endswith(".ckpt") :
            output_dir = Path(os.path.join(exp_path, experiment, output_name))
        else:
            output_dir = Path(os.path.join(exp_path, experiment.split('/')[0], output_name))
        output_dir.mkdir(exist_ok=True, parents=True)

    kwargs = {**kwargs, "callback": callback}
    save = Path(os.path.join(output_dir,"pred"))
    save.mkdir(exist_ok=True, parents=True)
    seed_everything(dataset.cfg.seed)

    loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)

    evaluate_single_image_demo(loader, model, multstage, save, **kwargs)

    return save