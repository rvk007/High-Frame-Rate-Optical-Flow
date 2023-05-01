import sys

sys.path.append('../core')
sys.path.append('/scratch/rr3937/optical_flow/raft/core')

import argparse
import os

import cv2
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from utils import flow_viz
from utils.utils import InputPadder

os.environ['WANDB_API_KEY']='7072ec59fb4f192caccd2cac53c77325644f637e'
wandb.login()
from raft import RAFT

DEVICE = 'cuda'


def generate_warped_image(flo, image1, image2):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    image1 = image1[0].permute(1,2,0).cpu().numpy()
    image2 = image2[0].permute(1,2,0).cpu().numpy()

    h, w = flo.shape[:2]
    flo[:,:,0] += np.arange(w)
    flo[:,:,1] += np.arange(h)[:,np.newaxis]
    warped_img = cv2.remap(image1, flo, None, cv2.INTER_LINEAR)
    residual = np.abs(warped_img - image2)
    residual = np.mean(residual, axis=2)

    return residual


def get_metrics(fps, flow_pred, flow_gt, im1, im2, th=3.0):
    photometric_error = np.mean(generate_warped_image(flow_pred, im1, im2))
    flow_pred = flow_pred[0].permute(1,2,0)
    flow_gt = torch.tensor(flow_gt).to(flow_pred.device)
    mae = sum( abs(flow_gt - flow_pred))
    epe = torch.sum((flow_pred - flow_gt)**2, dim=2).sqrt()

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'mae': mae.mean().item(),
        'pme': photometric_error,
    }
    wandb.log({
            f'val_epe/{fps}': metrics['epe'],
            f'val_1px/{fps}': metrics['1px'],
            f'val_3px/{fps}': metrics['3px'],
            f'val_5px/{fps}': metrics['5px'],
            f'val_mae/{fps}': metrics['mae'],
            f'val_pme/{fps}': metrics['pme'],
        })
    return metrics


def load_image(imfile):
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = np.array(cv2.imread(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# from BlenderProc
def flow_to_rgb(flow):
    """
    Visualizes optical flow in hsv space and converts it to rgb space.
    :param flow: (np.array (h, w, c)) optical flow
    :return: (np.array (h, w, c)) rgb data
    """
    im1 = flow[:, :, 0]
    im2 = flow[:, :, 1]
    h, w = flow.shape[:2]

    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 1] = 1

    mag, ang = cv2.cartToPolar(im1, im2)
    hsv[..., 0] = ang * 180 / np.pi
    hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def flow_visualization(fps, path, img1, img2, flo, flow_gt, idx
    ):

    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # get quiver plot of flow
    plot = False
    if plot:
        os.makedirs(f'quiver_{fps}', exist_ok=True)
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        ax = plt.subplot(gs[0])
        ax.imshow(img1)
        ax = plt.subplot(gs[1])
        ax.imshow(img2)
        ax = plt.subplot(gs[2])
        ax.imshow(flow_to_rgb(flo))
        ax = plt.subplot(gs[3])
        ax.quiver(flo[:, :, 0][::200], flo[:, :, 1][::200])
        plt.savefig(f'quiver_{fps}/plot_quiver_{idx}.jpg')


    # map flow to rgb image
    flo_viz= flow_viz.flow_to_image(flo) # correct RGB order
    flo_gt_viz = flow_viz.flow_to_image(flow_gt)
    # cv2.imwrite(f'{path}/raft/{idx}.jpg', flo_viz[:, :, [2,1,0]])

    # visualization of the results
    # npad = ((1, 1), (1, 1), (0, 0))
    # img1 = np.pad(img1, pad_width=npad, mode='constant', constant_values=0)
    # img2 = np.pad(img2, pad_width=npad, mode='constant', constant_values=0)
    # flo_viz = np.pad(flo_viz, pad_width=npad, mode='constant', constant_values=0)
    # flo_gt_viz = np.pad(flo_gt_viz, pad_width=npad, mode='constant', constant_values=0)
    # img = np.concatenate([img1, img2], axis=1)
    # flo = np.concatenate([flo_gt_viz, flo_viz], axis=1)
    # data = np.concatenate([img, flo], axis=0)
    cv2.imwrite(f'finetune_flow/no-weighting/results_fps_{fps}/{idx}.png', flo_viz[:,:,[2,1,0]])
    # image_data = wandb.Image(data, caption="FPS: %s" % fps)
    # wandb.log({
    #     f'Data_Fps_{fps}': image_data,
    # })
    # cv2.imwrite(f'{path}/raft/{idx}.jpg', data[:, :, [2,1,0]])

    # flow function from BelnderProc
    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(flow_to_rgb(flo))
    # plt.savefig(f'{path}/raft/{idx}.jpg')
    # flo_rgb = flow_to_rgb(flo)
    # cv2.imwrite(f'flo_{idx}.jpg', flo_rgb[:, :, [2,1,0]])


def plot_fps_vs_error(FPS, fps_metrics, path, cube_data):
    total_epe, total_1px, total_3px, total_5px, total_mae, total_pme = [], [], [], [], [], []
    for fps in FPS:
        epe, epe_1px, epe_3px, epe_5px, mae, pme = [], [], [], [], [], []
        for d in fps_metrics[fps]:
            epe.append(d['epe'])
            epe_1px.append(d['1px'])
            epe_3px.append(d['3px'])
            epe_5px.append(d['5px'])
            mae.append(d['mae'])
            pme.append(d['pme'])
        total_epe.append(np.mean(epe))
        total_1px.append(np.mean(epe_1px))
        total_3px.append(np.mean(epe_3px))
        total_5px.append(np.mean(epe_5px))
        total_mae.append(np.mean(mae))
        total_pme.append(np.mean(pme))
    line1, = plt.plot(FPS, total_epe, label='End point error')
    line2, = plt.plot(FPS, total_1px, label='End point error 1px')
    line3, = plt.plot(FPS, total_3px, label='End point error 3px')
    line4, = plt.plot(FPS, total_5px, label='End point error 5px')
    line5, = plt.plot(FPS, total_mae, label='Mean Absolute error')
    line6, = plt.plot(FPS, total_pme, label='Photometric error')

    plt.legend(handles=[line1, line2, line3, line4, line5, line6])
    plt.xlabel('FPS')
    plt.ylabel('Error')
    plt.savefig(f'{path}/{cube_data}_raft_error.jpg')


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    TIME_IN_SECONDS = 10
    OPTICAL_DATA_PATH = '/scratch/rr3937/optical_flow/dataset'
    if args.dev:
        FPS = [1]
    else:
        FPS = [1, 4, 8, 16, 24, 30, 60]
    DATA_FOLDER = {
        '1': 'cube_asphalt',
        '2': 'cube_brick',
        '3': 'cube_pavement',
        '4': 'sintel',
        '5': 'combined',
    }
    PATH = f'{OPTICAL_DATA_PATH}/{DATA_FOLDER[args.data]}'
    ckpt = args.model.split('/')[-1].split('.')[0]
    run = wandb.init(
        project="raft-no_fine_tuning-combined",
        name=DATA_FOLDER[args.data]+f'-{ckpt}',
        config=args,
        entity='rakhee998',
        mode='offline' if args.offline else 'online',
    )
    print(FPS)
    model = model.module
    model.to(DEVICE)
    model.eval()
    fps_metrics = {}

    for fps in FPS:
        os.makedirs(f'finetune_flow/no-weighting/results_fps_{fps}', exist_ok=True)
        all_metrics = []
        epe = []
        print(f'FPS: {fps}')
        path = f'{PATH}/{fps}/frame'
        os.makedirs(f'{path}/raft-{ckpt}', exist_ok=True)
        images = []
        for i in range(TIME_IN_SECONDS * fps):
            imfile = f'{path}/{i}.png'
            images.append(imfile)
        print(f'Running RAFT on {len(images)-1} images')

        with torch.no_grad():
            prev_frames = images[:-1]
            curr_frames = images[1:]
            i=0
            for imfile1, imfile2 in zip(prev_frames, curr_frames):
                if os.path.exists(imfile1) and os.path.exists(imfile2):
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)

                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    _, flow_up = model(image1, image2, iters=20, test_mode=True)

                    with h5py.File(f'{PATH}/{fps}/{i}.hdf5', 'r') as data:
                        flow_gt = data['forward_flow'][:]
                    if args.compute_metrics:
                        metrics = get_metrics(fps, flow_up, flow_gt, image1, image2)
                        all_metrics.append(metrics)
                        epe.append(metrics['epe'])

                    if args.visualize:
                        flow_visualization(fps, path, image1, image2, flow_up, flow_gt, i)
                    i += 1
                    # exit()
                else:
                    print(f'File {imfile1} or {imfile2} does not exist')
                    break
                # if args.dev:
                #     if i == 4:
                #         break
        fps_metrics[fps] = all_metrics
        print(f'FPS {fps}: {np.mean(epe)}')
        if args.compute_metrics:
            wandb.log({f'epe_mean/{fps}': np.mean(epe)})

    # plot a graph for fps vs epe
    plot_fps_vs_error(FPS, fps_metrics, PATH, DATA_FOLDER[args.data])
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--data', nargs='?')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--compute_metrics', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    demo(args)
