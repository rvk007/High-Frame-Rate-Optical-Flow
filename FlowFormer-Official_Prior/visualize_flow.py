import sys

sys.path.append('core')

import argparse
import math
import os
import os.path as osp
from glob import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from configs.submission import get_cfg
from core.FlowFormer import build_flowformer
from core.utils.misc import process_cfg
from utils import flow_viz, frame_utils
from utils.utils import InputPadder, forward_interpolate

os.environ['WANDB_API_KEY']='7072ec59fb4f192caccd2cac53c77325644f637e'
wandb.login()
# TRAIN_SIZE = [432, 960]
TRAIN_SIZE = [1024, 1024]
parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='?')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--compute_metrics', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--eval_type', default='seq')
parser.add_argument('--path', default='.')
parser.add_argument('--sintel_dir', default='datasets/Sintel/test/clean')
parser.add_argument('--seq_dir', default='demo_data/mihoyo')
parser.add_argument('--start_idx', type=int, default=1)     # starting index of the image sequence
parser.add_argument('--end_idx', type=int, default=1200)    # ending index of the image sequence
parser.add_argument('--keep_size', action='store_true')     # keep the image size, or the image will be adaptively resized.
args = parser.parse_args()

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


def center_crop(data, shape):
    if not (0 < shape[0] <= data.shape[0] and 0 < shape[1] <= data.shape[1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[0] - shape[0]) // 2
    h_from = (data.shape[1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[w_from:w_to, h_from:h_to,:]


def generate_warped_image(flo, image1, image2):
    h, w = flo.shape[:2]
    flo[:,:,0] += np.arange(w)
    flo[:,:,1] += np.arange(h)[:,np.newaxis]
    warped_img = cv2.remap(image1, flo, None, cv2.INTER_LINEAR)
    residual = np.abs(warped_img - image2)
    residual = np.mean(residual, axis=2)

    return residual


def get_metrics(fps, flow_pred, flow_gt, im1, im2, th=3.0):
    im1 = im1.permute(1,2,0).cpu().numpy()
    im2 = im2.permute(1,2,0).cpu().numpy()
    # im1 = center_crop(im1, [960,960])
    # im2 = center_crop(im2, [960,960])
    # flow_gt = center_crop(flow_gt, [960,960])
    photometric_error = np.mean(generate_warped_image(flow_pred, im1, im2))
    flow_gt = torch.tensor(flow_gt)
    epe = torch.norm(flow_gt - flow_pred, p=2, dim=1)
    mae = sum( abs(flow_gt - flow_pred))

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


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights


def compute_flow(model, image1, image2, weights=None):
    # print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1]

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size


def prepare_image(root_dir, fn1, fn2, keep_size):
    # print(f"preparing image...")
    # print(f"root dir = {root_dir}, fn = {fn1}")

    image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


    # dirname = osp.dirname(fn1)
    # filename = osp.splitext(osp.basename(fn1))[0]

    # viz_dir = osp.join(viz_root_dir, dirname)
    # if not osp.exists(viz_dir):
    #     os.makedirs(viz_dir)

    # viz_fn = osp.join(viz_dir, filename + '.png')

    return image1, image2

def build_model():
    print(f"Building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

def visualize_flow(root_dir, model, images, keep_size):
    weights = None
    prev_frames = images[:-1]
    curr_frames = images[1:]
    idx = 0
    all_metrics = []
    for imfile1, imfile2 in zip(prev_frames, curr_frames):
        fn1, fn2 = imfile1, imfile2
        image1, image2 = prepare_image(root_dir, fn1, fn2, keep_size)

        flow = compute_flow(model, image1, image2, weights)
        with h5py.File(f'{root_dir}/{fps}/{i}.hdf5', 'r') as data:
            flow_gt = data['forward_flow'][:]
        if args.compute_metrics:
            metrics = get_metrics(fps, flow, flow_gt, image1, image2)
            all_metrics.append(metrics)
        if args.visualize:
            flow_img = flow_viz.flow_to_image(flow)
            flow_gt_img = flow_viz.flow_to_image(flow_gt)
            # print(image1.shape, image2.shape, flow_img.shape, flow_gt_img.shape)
            # visualization of the results
            npad = ((1, 1), (1, 1), (0, 0))
            image1 = np.pad(image1.squeeze(0).permute(1,2,0).cpu().numpy(), pad_width=npad, mode='constant', constant_values=0)
            image2 = np.pad(image2.squeeze(0).permute(1,2,0).cpu().numpy(), pad_width=npad, mode='constant', constant_values=0)
            flow_img = np.pad(flow_img, pad_width=npad, mode='constant', constant_values=0)
            flow_gt_img = np.pad(flow_gt_img, pad_width=npad, mode='constant', constant_values=0)
            # print(image1.shape, image2.shape, flow_img.shape, flow_gt_img.shape)

            img = np.concatenate([image1, image2], axis=1)
            flo = np.concatenate([flow_gt_img, flow_img], axis=1)
            data = np.concatenate([img, flo], axis=0)
            image_data = wandb.Image(data, caption="FPS: %s" % fps)
            wandb.log({
                f'Data_Fps_{fps}': image_data,
            })


            # flow_img = flow_to_rgb(flow)
            # cv2.imwrite(f'{root_dir}/{fps}/frame/flowformer_n/{idx}.jpg', flow_img[:, :, [2,1,0]])
        idx += 1
        # if args.dev:
        #     if idx == 4:
        #         break
    return all_metrics


def process_sintel(sintel_dir):
    img_pairs = []
    for scene in os.listdir(sintel_dir):
        dirname = osp.join(sintel_dir, scene)
        image_list = sorted(glob(osp.join(dirname, '*.png')))
        for i in range(len(image_list)-1):
            img_pairs.append((image_list[i], image_list[i+1]))

    return img_pairs


def generate_pairs(dirname, start_idx, end_idx):
    img_pairs = []
    for idx in range(start_idx, end_idx):
        img1 = osp.join(dirname, f'{idx:06}.png')
        img2 = osp.join(dirname, f'{idx+1:06}.png')
        img_pairs.append((img1, img2))

    return img_pairs


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
    plt.savefig(f'{path}/{cube_data}_flowformer_error.jpg')


if __name__ == '__main__':
    fps_metrics = {}
    run = wandb.init(
        project="ff-no_fine_tuning-combined",
        name=DATA_FOLDER[args.data],
        config=args,
        entity='rakhee998'
    )
    print(f"Running on {DATA_FOLDER[args.data]}")
    print(args)
    print(FPS)
    model = build_model()
    for fps in FPS:
        print(f'FPS: {fps}')
        os.makedirs(f'{PATH}/{fps}/frame/flowformer_n', exist_ok=True)
        images = []
        for i in range(TIME_IN_SECONDS * fps):
            imfile = f'{PATH}/{fps}/frame/{i}.png'
            images.append(imfile)
        print(f'Number of images: {len(images)}')
        with torch.no_grad():
            all_metrics= visualize_flow(PATH, model, images, args.keep_size)
            fps_metrics[fps] = all_metrics

    plot_fps_vs_error(FPS, fps_metrics, PATH, DATA_FOLDER[args.data])
