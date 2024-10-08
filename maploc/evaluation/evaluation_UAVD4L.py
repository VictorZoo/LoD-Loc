# # Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from OrienterNet, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/facebookresearch/OrienterNet
# Released under the Apache License 2.0, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import OmegaConf, DictConfig

from .. import logger, DATASETS_PATH
from ..conf import data as conf_data_dir
from ..data import UAVDataModule
from .run import evaluate

data_dir =  DATASETS_PATH / "UAVD4L-LoD"

model_overrides = {
    "model": {
        "name": 'LoD_Loc',
        "lamb_val": [0.8,0.8,0.8],
        "num_sample_val": [[8, 10, 10, 30],[8, 10, 10, 30],[8, 10, 10, 30]]
    },
}


data_cfg_train = OmegaConf.load(Path(conf_data_dir.__file__).parent / "UAVD4L-LoD.yaml")
data_cfg = OmegaConf.merge(
    data_cfg_train,
    {
        "loading": {"val": {"batch_size": 1, "num_workers": 0, "num_sample":2000}},
    },
)
default_cfg_single = OmegaConf.create({"data": data_cfg})


def run(
    split: str,
    experiment: str,
    multstage: int,
    cfg: Optional[DictConfig] = None,
    thresholds: Tuple[int] = (1, 3, 5, 7),
    **kwargs,
):
    cfg = cfg or {}
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    default = default_cfg_single

    default = OmegaConf.merge(default, dict(model_overrides))
    cfg = OmegaConf.merge(default, cfg)
    dataset = UAVDataModule(cfg.get("data", {}))

    metrics = evaluate(experiment, cfg, dataset, split, multstage=multstage, **kwargs)

    keys = [
        "xyz_error",
        "yaw_error",
    ]
    for k in keys:
        if k not in metrics:
            logger.warning("Key %s not in metrics.", k)
            continue
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        logger.info("Recall %s: %s at %s m/°", k, rec, thresholds)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--multistage", type=int, default=0 )
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--num", type=int)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.from_cli(args.dotlist)
    run(
        args.split,
        args.experiment,
        args.multistage,
        cfg,
        output_name=args.output_name
    )
