import sys

sys.path.append('core')

import argparse
import glob
import os
import math

import cv2
import numpy as np
import torch
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from flow_utils import generate_warped_image, get_color_wheel_distance

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def flow_visualization(img1, img2, flo, idx, model_name, plot=False,
                        compute_distance=False, warp=False
    ):
    combine_img_flow = False
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo_rgb = flow_viz.flow_to_image(flo) # correct RGB order

    if combine_img_flow:
        img_flo = np.concatenate([img1, flo_rgb], axis=0)
        if 'kitti' in model_name:
            folder_name = 'out_kitti'
        elif 'sintel' in model_name:
            folder_name = 'out_sintel'
        elif 'chairs' in model_name:
            folder_name = 'out_chairs'
        elif 'things' in model_name:
            folder_name = 'out_things'
        elif 'small' in model_name:
            folder_name = 'out_small'
        filename = f'{folder_name}/{idx}.png'
        # cv2.imwrite(f'img_flo/{idx}.jpg', img_flo[:, :, [2,1,0]])

    if warp:
        # print('Generating warped image...')
        warped_img, residual = generate_warped_image(flo, img1, img2, idx)
        photometric_error = np.mean(residual)
        # combine_pred_target = np.concatenate([img2, warped_img], axis=0)
        cv2.imwrite(f'warp/{idx}.jpg', warped_img)
        cv2.imwrite(f'residual/{idx}.jpg', residual)

    if compute_distance:
        flo_dist = get_color_wheel_distance(flo_rgb, idx)

    if plot:
        pme = str(round(photometric_error,2))
        # save_plot(img1, img2, flo_dist, warped_img, residual, pme, idx)
        cv2.imwrite(f'flow/{idx}.jpg', flo_dist)

    return pme


def save_plot(img1, img2, flo_dist, warped_img, residual, photometric_error, idx):
    # print('Saving plot...')

    plt.figure(figsize = (5,1))
    plt.rcParams['font.size'] = 5
    gs1 = gridspec.GridSpec(1,5)
    gs1.update(wspace=0.35, hspace=0.25) # set the spacing between axes.
    to_plot = [img1, img2, warped_img, residual, flo_dist[:,:,[2,1,0]]]
    plot_labels = ['Previous Frame', 'Current Frame', 'Warped Frame', \
        f'Residual\n[Photometric\nError: {photometric_error}]', 'Flow visualization'
    ]
    for i in range(5):
        ax = plt.subplot(gs1[i])
        ax.imshow(to_plot[i]/255.0)
        ax.set_title(plot_labels[i])
        ax.axis('off')
        ax.set_aspect('equal')

    plt.savefig(f'results/{idx}.jpg', dpi=480, bbox_inches='tight', pad_inches=0.1)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        prev_frames = images[:-1]
        curr_frames = images[1:]
        i=0
        photometric_error = []
        # i = int(args.start)
        # end = int(args.end)
        # if i == 0:
        #     prev_frames = images[i:end]
        #     curr_frames = images[i+1:end+1]
        # else:
        #     prev_frames = images[i:-1]
        #     curr_frames = images[i+1:]

        # image1 = load_image(images[i])
        # image2 = load_image(images[i+1])

        # padder = InputPadder(image1.shape)
        # image1, image2 = padder.pad(image1, image2)

        # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        # flow_visualization(image1, image2, flow_up, i, model_name=args.model,
        #     plot=True, compute_distance=True, warp=True
        # )
        for imfile1, imfile2 in zip(prev_frames, curr_frames):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            pme = flow_visualization(image1, image2, flow_up, i, model_name=args.model,
                plot=True, compute_distance=True, warp=True
            )
            photometric_error.append(pme)
            i += 1
        print(f'Average photometric error: {np.mean(photometric_error)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    # parser.add_argument('--start', help="restore checkpoint")
    # parser.add_argument('--end', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
