import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time
import logging
import itertools

from det3d.utils.config_tool import get_downsample_factor

CHECK_POINT = "/workspace/data/voxelnet_converted.pth"
CONFIG_FILE = "/workspace/code/CenterPoint/det3d/test_config.py"

def main():
    # config
    cfg = Config.fromfile(CONFIG_FILE)
    # model loading
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, CHECK_POINT, map_location="cpu")
    model = model.cuda()
    model.eval()
    # data loader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    # infer
    detections = []
    for i, data_batch in enumerate(data_loader):
        print ("step:", i)
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=0,
            )
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
    all_predictions = all_gather(detections)

    
if __name__ == "__main__":
    main()

