# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
import argparse
import glob
import os
from multiprocessing import Pool

import cv2 as cv
import numpy as np
import torch
from PIL import Image

print('Running Lucas-Kanade')
parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='?')
parser.add_argument('--dev', action='store_true')
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
    flo = flo[0].permute(1,2,0).cpu().numpy()
    image1 = image1[0].permute(1,2,0).cpu().numpy()
    image2 = image2[0].permute(1,2,0).cpu().numpy()

    h, w = flo.shape[:2]
    flo[:,:,0] += np.arange(w)
    flo[:,:,1] += np.arange(h)[:,np.newaxis]
    warped_img = cv.remap(image1, flo, None, cv.INTER_LINEAR)
    residual = np.abs(warped_img - image2)
    residual = np.mean(residual, axis=2)
    return residual


def get_metrics(flow_pred, flow_gt, im1, im2, th=3.0):
    photometric_error = np.mean(generate_warped_image(flow_pred, im1, im2))
    flow_pred = flow_pred[0].permute(1,2,0)
    flow_gt = torch.tensor(flow_gt).to(flow_pred.device)
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


def lucas_kanade(fps, path):
    print(f'path: {path}')
    images = glob.glob(os.path.join(path, '*.png')) + \
                 glob.glob(os.path.join(path, '*.jpg'))

    images = sorted(images)
    print('Performing Lucas-Kanade on {} images'.format(len(images)))
    prev_frames = images[:-1]
    curr_frames = images[1:]

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    idx = 0
    for imfile1, imfile2 in zip(prev_frames, curr_frames):
        image1 = cv.imread(imfile1)
        image2 = cv.imread(imfile2)

        old_frame = image1
        frame = image2

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        print(f'p1: {p1.shape}')
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imwrite(f'{path}/lucas_kanade/{idx}.jpg', img)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        idx += 1


# using multiprocessing
for fps in FPS:
    print(f'fps: {fps}')
    path = f'{PATH}/{fps}/frame'
    os.makedirs(f'{PATH}/{fps}/frame/lucas_kanade', exist_ok=True)
    with Pool(8) as p:
        p.starmap(lucas_kanade, [(fps, path)])

print('Done')
