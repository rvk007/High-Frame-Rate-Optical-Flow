import argparse
import itertools
import math
import os
from multiprocessing import Pool

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PerceiverForOpticalFlow

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='?')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--compute_metrics', action='store_true')
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

TIME_IN_SECONDS = 10
OPTICAL_DATA_PATH = '/scratch/rr3937/optical_flow/dataset'
if args.dev:
    FPS = [1, 4, 8, 16, 24]
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")
model.to(device)
TRAIN_SIZE = model.config.train_size


def get_metrics(flow_pred, flow_gt):
    flow_gt = torch.tensor(flow_gt).to(device)
    flow_pred = torch.tensor(flow_pred).to(device)
    epe = torch.norm(flow_gt - flow_pred, p=2, dim=1)
    mae = sum( abs(flow_gt - flow_pred))

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'mae': mae.mean().item(),
    }
    return metrics


def normalize(im):
  return im / 255.0 * 2 - 1


# source: https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/9
def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))

    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()

    return patches.view(b,-1,patches.shape[-2], patches.shape[-1])


def compute_optical_flow(model, img1, img2, grid_indices, FLOW_SCALE_FACTOR = 20):
  """Function to compute optical flow between two images.

  To compute the flow between images of arbitrary sizes, we divide the image
  into patches, compute the flow for each patch, and stitch the flows together.

  Args:
    model: PyTorch Perceiver model
    img1: first image
    img2: second image
    grid_indices: indices of the upper left corner for each patch.
  """
  img1 = torch.tensor(np.moveaxis(img1, -1, 0))
  img2 = torch.tensor(np.moveaxis(img2, -1, 0))
  imgs = torch.stack([img1, img2], dim=0)[None]
  height = imgs.shape[-2]
  width = imgs.shape[-1]

  #print("Shape of imgs after stacking:", imgs.shape)
  patch_size = model.config.train_size

  if height < patch_size[0]:
    raise ValueError(
        f"Height of image (shape: {imgs.shape}) must be at least {patch_size[0]}."
        "Please pad or resize your image to the minimum dimension."
    )
  if width < patch_size[1]:
    raise ValueError(
        f"Width of image (shape: {imgs.shape}) must be at least {patch_size[1]}."
        "Please pad or resize your image to the minimum dimension."
    )

  flows = 0
  flow_count = 0

  for y, x in grid_indices:
    imgs = torch.stack([img1, img2], dim=0)[None]
    inp_piece = imgs[..., y : y + patch_size[0],
                     x : x + patch_size[1]]

    # print("Shape of inp_piece:", inp_piece.shape)

    batch_size, _, C, H, W = inp_piece.shape
    patches = extract_image_patches(inp_piece.view(batch_size*2,C,H,W), kernel=3)
    _, C, H, W = patches.shape
    patches = patches.view(batch_size, -1, C, H, W).float().to(model.device)

    # actual forward pass
    with torch.no_grad():
      output = model(inputs=patches).logits * FLOW_SCALE_FACTOR

    # the code below could also be implemented in PyTorch
    flow_piece = output.cpu().detach().numpy()

    weights_x, weights_y = np.meshgrid(
        torch.arange(patch_size[1]), torch.arange(patch_size[0]))

    weights_x = np.minimum(weights_x + 1, patch_size[1] - weights_x)
    weights_y = np.minimum(weights_y + 1, patch_size[0] - weights_y)
    weights = np.minimum(weights_x, weights_y)[np.newaxis, :, :,
                                                np.newaxis]
    padding = [(0, 0), (y, height - y - patch_size[0]),
               (x, width - x - patch_size[1]), (0, 0)]
    flows += np.pad(flow_piece * weights, padding)
    flow_count += np.pad(weights, padding)

    # delete activations to avoid OOM
    del output

  flows /= flow_count
  return flows


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  ys = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  xs = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))
  # Make sure the final patch is flush with the image boundary
  ys[-1] = image_shape[0] - patch_size[0]
  xs[-1] = image_shape[1] - patch_size[1]
  return itertools.product(ys, xs)


def visualize_flow(path, idx, flow):
  flow = np.array(flow)
  # Use Hue, Saturation, Value colour model
  hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
  hsv[..., 2] = 255

  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang / np.pi / 2 * 180
  hsv[..., 1] = np.clip(mag * 255 / 24, 0, 255)
  bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  plt.imshow(bgr)
  plt.savefig(f'{path}/perceiver/{idx}.jpg')


def perceiver_optical_flow(path, fps):
  images = []
  all_metrics = []
  for i in range(TIME_IN_SECONDS * fps):
      imfile = f'{path}/{i}.png'
      images.append(imfile)
  prev_frames = images[:-1]
  curr_frames = images[1:]
  idx = 0
  for imfile1, imfile2 in zip(prev_frames, curr_frames):
    image1 = cv2.imread(imfile1)
    image2 = cv2.imread(imfile2)
    im1 = np.array(image1)
    im2 = np.array(image2)
    # Divide images into patches, compute flow between corresponding patches
    # of both images, and stitch the flows together
    grid_indices = compute_grid_indices(im1.shape)
    flow = compute_optical_flow(model, normalize(im1), normalize(im2), grid_indices)
    if args.compute_metrics:
      with h5py.File(f'{args.path}/{fps}/{i}.hdf5', 'r') as data:
        flow_gt = data['forward_flow'][:]
      metrics = get_metrics(flow, flow_gt, image1, image2)
      all_metrics.append(metrics)
    if args.visualize:
      visualize_flow(path, idx, flow[0])
    idx += 1
    if args.dev:
      if i == 4:
          break
  return all_metrics


def plot_fps_vs_error(FPS, fps_metrics, path, cube_data):
    total_epe, total_1px, total_3px, total_5px, total_mae = [], [], [], [], []
    for fps in FPS:
        epe, epe_1px, epe_3px, epe_5px, mae = [], [], [], [], []
        for d in fps_metrics[fps]:
            epe.append(d['epe'])
            epe_1px.append(d['1px'])
            epe_3px.append(d['3px'])
            epe_5px.append(d['5px'])
            mae.append(d['mae'])
        total_epe.append(np.mean(epe))
        total_1px.append(np.mean(epe_1px))
        total_3px.append(np.mean(epe_3px))
        total_5px.append(np.mean(epe_5px))
        total_mae.append(np.mean(mae))

    line1, = plt.plot(FPS, total_epe, label='End point error')
    line2, = plt.plot(FPS, total_1px, label='End point error 1px')
    line3, = plt.plot(FPS, total_3px, label='End point error 3px')
    line4, = plt.plot(FPS, total_5px, label='End point error')
    line5, = plt.plot(FPS, total_mae, label='Mean Absolute error')

    plt.legend(handles=[line1, line2, line3, line4, line5,])
    plt.xlabel('FPS')
    plt.ylabel('Error')
    plt.savefig(f'{path}/{cube_data}_perceiver_error.jpg')


# using multiprocessing
fps_metrics = {}
for fps in FPS:
    print(f'fps: {fps}')
    path = f'{PATH}/{fps}/frame'
    os.makedirs(f'{PATH}/{fps}/frame/perceiver', exist_ok=True)
    all_metrics = perceiver_optical_flow(path, fps)
    fps_metrics[fps] = all_metrics
    # with Pool(8) as p:
    #     p.starmap(perceiver_optical_flow, [(path, fps)])

plot_fps_vs_error(FPS, fps_metrics, PATH, DATA_FOLDER[args.data])
print('Done')
