import sys

sys.path.append('core')

import argparse
import os
import time

import cv2
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from configs.default import get_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg
from configs.things_eval import get_cfg as get_things_cfg

# from FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer
from core.utils.misc import process_cfg
from PIL import Image
from utils import flow_viz, frame_utils
from utils.utils import InputPadder, forward_interpolate

from raft import RAFT


@torch.no_grad()
def validate_chairs(model):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        flow_pre, _ = model(image1, image2)

        epe = torch.sum((flow_pre[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_combined(model):
    """ Peform validation using the Combined (validation) split """
    model.eval()
    results = {}
    for fps in [1, 4, 8, 16, 24, 30, 60]:
        val_dataset = datasets.SimulatedCombined(split='validation', fps=fps)
        epe_list = []
        os.makedirs(f'basicencoder/results_fps_{fps}', exist_ok=True)
        for val_id in range(len(val_dataset)):
            imgs, flow_gt, _= val_dataset[val_id]
            imgs = [img[None].cuda() for img in imgs]
            image1, image2, image3, image4 = imgs[0], imgs[1], imgs[2], imgs[3]
            padder = InputPadder(image1.shape)
            image1, image2, image3, image4 = padder.pad(image1, image2, image3, image4)

            flow_pre = model([image1, image2, image3, image4])

            flow_pre = padder.unpad(flow_pre[0]).cpu()[0]

            # epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            # epe_list.append(epe.view(-1).numpy())

            # visualization of the results
            # npad = ((1, 1), (1, 1), (0, 0))
            # image3 = np.pad(image3.squeeze(0).permute(1,2,0).cpu().numpy(), pad_width=npad, mode='constant', constant_values=0)
            # image4 = np.pad(image4.squeeze(0).permute(1,2,0).cpu().numpy(), pad_width=npad, mode='constant', constant_values=0)
            # flow_gt = flow_gt.permute(1,2,0).cpu().numpy()
            # flow_pre = flow_pre.permute(1,2,0).cpu().numpy()

            # flow_gt_viz = np.pad(flow_viz.flow_to_image(flow_gt), pad_width=npad, mode='constant', constant_values=0)
            # flow_pre_viz = np.pad(flow_viz.flow_to_image(flow_pre), pad_width=npad, mode='constant', constant_values=0)
            # img = np.concatenate([image3, image4], axis=1)
            # flo = np.concatenate([flow_gt_viz, flow_pre_viz], axis=1)
            # data = np.concatenate([img, flo], axis=0)
            flow_pre = flow_pre.permute(1,2,0).cpu().numpy()
            flow_pred = flow_viz.flow_to_image(flow_pre)
            cv2.imwrite(f'basicencoder/results_fps_{fps}/{val_id}.png', flow_pred[:,:,[2,1,0]])
            # image_data = wandb.Image(data, caption="FPS: %s" % fps)
            # wandb.log({
            #     f'Data_Fps_{fps}': image_data,
            # })
        # epe_all = np.concatenate(epe_list)
        # epe = np.mean(epe_all)
        # px1 = np.mean(epe_all<1)
        # px3 = np.mean(epe_all<3)
        # px5 = np.mean(epe_all<5)
        # wandb.log({
        #     f'val_epe/{fps}': epe,
        #     f'val_1px/{fps}': px1,
        #     f'val_3px/{fps}': px3,
        #     f'val_5px/{fps}': px5,
        # })

        # print("Validation FPS:(%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (fps, epe, px1, px3, px5))
        # results[fps] = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_sintel(model):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pre = model(image1, image2)

            flow_pre = padder.unpad(flow_pre[0]).cpu()[0]

            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results

@torch.no_grad()
def create_sintel_submission(model, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """

    model.eval()
    for dstype in ['final', "clean"]:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1, image2 = image1[None].cuda(), image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pre = model(image1, image2)

            flow_pre = padder.unpad(flow_pre[0]).cpu()
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)


@torch.no_grad()
def validate_kitti(model):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre = model(image1, image2)

        flow_pre = padder.unpad(flow_pre[0]).cpu()[0]

        epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--nw', action='store_true')
    parser.add_argument('--use_svt', action='store_true')
    args = parser.parse_args()

    exp_name = 'combined-2200_ff-combined_no_weighting_svt' if args.use_svt else 'combined-1000_ff-combined_no_weighting_basicdecoder'
    run = wandb.init(
        project="ff-no_fine_tuning-combined",
        name = exp_name,
        config=args,
        entity='rakhee998',
        mode = 'offline' if args.offline else 'online'
    )

    # cfg = get_cfg()
    if args.small:
        cfg = get_small_things_cfg()
    if args.nw:
        from configs.sintel import get_cfg
        cfg = get_cfg()
    else:
        cfg = get_things_cfg()

    # svt
    if args.use_svt:
        args.model = '/scratch/rr3937/optical_flow/FlowFormer-Official_Prior/logs/ff-combined/latentcostformer/no_weighting/cost_heads_num[1]vert_c_dim[64]cnet[twins]pretrain[True]add_flow_token[True]encoder_depth[3]gma[GMA]cost_encoder_res[True]sintel(04_30_03_53)/2000_ff-combined.pth'
        args.use_prior = 'svt'
    else:
        args.model = '/scratch/rr3937/optical_flow/FlowFormer-Official_Prior/logs/ff-combined/latentcostformer/no_weighting/cost_heads_num[1]vert_c_dim[64]cnet[twins]pretrain[True]add_flow_token[True]encoder_depth[3]gma[GMA]cost_encoder_res[True]sintel(04_30_14_01)/1000_ff-combined.pth'
        args.use_prior = 'basicencoder'

    print(exp_name)
    print(args.model)
    print(args.use_prior)
    cfg.update(vars(args))
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    print(args)

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'combined':
            validate_combined(model.module)

        elif args.dataset == 'sintel_submission':
            create_sintel_submission(model.module)


# conda activate craft

# FILE=/scratch/rr3937/optical_flow/FlowFormer-Official_Prior/evaluate_FlowFormer.py
# python ${FILE} --dataset combined --model /scratch/rr3937/optical_flow/FlowFormer-Official/checkpoints/sintel.pth
