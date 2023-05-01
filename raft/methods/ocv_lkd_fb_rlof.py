# https://gist.github.com/FingerRec/eba088d6d7a50e17c875d74684ec2849
import argparse
import glob
import os
from multiprocessing import Pool

import cv2 as cv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='?')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--compute_metrics', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument(
        "--algorithm",
        choices=["farneback", "lucaskanade_dense", "rlof"],
        required=True,
        help="Optical flow algorithm to use",
    )
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


def generate_warped_image(flo, image1, image2):
    h, w = flo.shape[:2]
    flo[:,:,0] += np.arange(w)
    flo[:,:,1] += np.arange(h)[:,np.newaxis]
    warped_img = cv.remap(image1, flo, None, cv.INTER_LINEAR)
    residual = np.abs(warped_img - image2)
    residual = np.mean(residual, axis=2)

    return residual

def get_metrics(flow_pred, flow_gt, im1, im2, th=3.0):
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
    return metrics


def dense_optical_flow(main_path, algorithm, method, fps, path, params=[], to_gray=False):
    images = glob.glob(os.path.join(path, '*.png')) + \
                 glob.glob(os.path.join(path, '*.jpg'))

    images = sorted(images)
    print(f'Performing {algorithm} on {len(images)} images')
    prev_frames = images[:-1]
    curr_frames = images[1:]

    idx = 0
    all_metrics = []
    for imfile1, imfile2 in zip(prev_frames, curr_frames):
        image1 = cv.imread(imfile1)
        image2 = cv.imread(imfile2)

        old_frame = image1
        frame = image2

        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255

        if to_gray:
            old_frame = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # calculate Optical Flow
        flow = method(old_frame, frame, None, *params)

        flow /= 10
        if args.compute_metrics:
            if os.path.exists(f'{main_path}/{fps}/{idx}.hdf5'):
                with h5py.File(f'{main_path}/{fps}/{idx}.hdf5', 'r') as data:
                    flow_gt = data['forward_flow'][:]
                metrics = get_metrics(flow, flow_gt, image1, image2)
                all_metrics.append(metrics)
        if args.visualize:
            # Encoding: convert the algorithm's output into Polar coordinates
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            # Use Hue and Saturation to encode the Optical Flow
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            # Convert HSV image into BGR for demo
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imwrite(f'{path}/{algorithm}/{idx}.jpg', bgr)
        old_frame = frame
        idx += 1
        if args.dev:
            if idx == 4:
                break
    return all_metrics



def ocv_algorithm(PATH, algorithm, fps, frame_path):
    print(PATH, algorithm, fps, frame_path)
    os.makedirs(f'{frame_path}/{algorithm}', exist_ok=True)
    if algorithm == "lucaskanade_dense":
        print('Running Lucas-Kanade Dense')
        method = cv.optflow.calcOpticalFlowSparseToDense
        all_metrics = dense_optical_flow(
            PATH, algorithm, method, fps, frame_path, to_gray=True
        )
    elif algorithm == "farneback":
        print('Running Farneback')
        method = cv.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
        all_metrics = dense_optical_flow(
            PATH, algorithm, method, fps, frame_path, params, to_gray=True
        )
    elif algorithm == "rlof":
        print('Running RLOF')
        method = cv.optflow.calcOpticalFlowDenseRLOF
        all_metrics = dense_optical_flow(
            PATH, algorithm, method, fps, frame_path
        )
    return all_metrics


def plot_fps_vs_error(FPS, fps_metrics, path, cube_data, algorithm):
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
    plt.savefig(f'{path}/{cube_data}_{algorithm}_error.jpg')


# using multiprocessing
fps_metrics = {}
for fps in FPS:
    all_metrics = []
    print(f'fps: {fps}')
    frame_path = f'{PATH}/{fps}/frame'
    # all_metrics = ocv_algorithm(PATH, args.algorithm, fps, frame_path)
    # fps_metrics[fps] = all_metrics
    with Pool(8) as p:
        p.starmap(ocv_algorithm, [(PATH, args.algorithm, fps, frame_path)])

# plot_fps_vs_error(FPS, fps_metrics, PATH, DATA_FOLDER[args.data], args.algorithm)
