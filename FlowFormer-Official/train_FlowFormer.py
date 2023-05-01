from __future__ import division, print_function

import argparse
import os
import sys
import time
from pathlib import Path

import core.datasets as datasets
import cv2
import evaluate_FlowFormer as evaluate
import evaluate_FlowFormer_tile as evaluate_tile
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core import optimizer

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

#torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(cfg):
    model = nn.DataParallel(build_flowformer(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    print('train_loader loaded')
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    add_noise = False

    should_keep_training = True
    while should_keep_training:
        print(f'Start training: {len(train_loader)}')
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            imgs, flow, valid = [x for x in data_blob]
            imgs = [x.cuda() for x in imgs] # 3 * 6 * [3, 432, 960] -> 6: Batch size, 3: Number of images
            flow = flow.cuda()
            valid = valid.cuda()


            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                imgs = [(x + stdv * torch.randn(*x.shape).cuda()).clamp(0.0, 255.0) for x in imgs]
                # image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                # image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            output = {}
            flow_predictions = model(imgs, output)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)

            ### change evaluate to functions

            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                # torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)

                model.train()

            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    PATH = f'checkpoints/{cfg.stage}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowformer', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    args = parser.parse_args()

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'autoflow':
        from configs.autoflow import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(cfg)
