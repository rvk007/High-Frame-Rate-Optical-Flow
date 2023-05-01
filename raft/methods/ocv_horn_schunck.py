# https://github.com/lmiz100/Optical-flow-Horn-Schunck-method/blob/master/MyHornSchunck.py
import glob
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as filter2

TIME_IN_SECONDS = 10
parser = ArgumentParser(description = 'Horn Schunck program')
parser.add_argument('--data', nargs='?')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--compute_metrics', action='store_true')
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

TIME_IN_SECONDS = 10
OPTICAL_DATA_PATH = '/scratch/rr3937/optical_flow/dataset'
if args.dev:
    FPS = [1, 4, 8, 16]
else:
    FPS = [24, 30, 60]
DATA_FOLDER = {
    '1': 'cube_asphalt',
    '2': 'cube_brick',
    '3': 'cube_pavement',
    '4': 'sintel',
    '5': 'combined',
}
PATH = f'{OPTICAL_DATA_PATH}/{DATA_FOLDER[args.data]}'

def get_metrics(flow_pred, flow_gt):
    flow_gt = torch.tensor(flow_gt)
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

def show_image(name, image):
    if image is None:
        return

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg

def draw_quiver(u, v, beforeImg, path, idx):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.savefig(f'{path}/horn_schunck/{idx}.png')

def get_derivatives(img1, img2):
    #derivative masks
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1,x_kernel) + filter2(img2,x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx,fy, ft]

def computeHS(root_dir, name1, name2, alpha, delta, path, idx):
    # path = os.path.join(os.path.dirname(__file__), 'test images')
    beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE)
    afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE)

    if beforeImg is None:
        raise NameError("Can't find image: \"" + name1 + '\"')
    elif afterImg is None:
        raise NameError("Can't find image: \"" + name2 + '\"')

    beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE).astype(float)
    afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE).astype(float)

    im1 = cv2.imread(os.path.join(path, name1))
    im2 = cv2.imread(os.path.join(path, name2))

    #removing noise
    beforeImg  = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)

    # set up initial values
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = get_derivatives(beforeImg, afterImg)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d = 4 * alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        #converges check (at most 300 iterations)
        if  diff < delta or iter_counter > 300:
            # print("iteration number: ", iter_counter)
            break

    # draw_quiver(u, v, beforeImg, path, idx)
    h, w = u.shape

    metrics = {}
    flow = np.dstack((u, v))
    if args.compute_metrics:
        if os.path.exists(f'{root_dir}/{fps}/{idx}.hdf5'):
            with h5py.File(f'{root_dir}/{fps}/{idx}.hdf5', 'r') as data:
                flow_gt = data['forward_flow'][:]
            metrics = get_metrics(flow, flow_gt, im1, im2)
            return metrics
    if args.visualize:
        # Use Hue, Saturation, Value colour model
        hsv = np.zeros((h, w, 3), dtype=np.float32)
        hsv[..., 1] = 1

        mag, ang = cv2.cartToPolar(u, v)
        hsv[..., 0] = ang * 180 / np.pi
        hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

        plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        plt.savefig(f'{path}/horn_schunck/{idx}.jpg')


def horn_schunck(p, fps, path):
    all_metrics = []
    images = glob.glob(os.path.join(path, '*.png')) + \
                 glob.glob(os.path.join(path, '*.jpg'))

    images = sorted(images)
    print(f'Performing Horn Schunck on {len(images)} images')
    prev_frames = images[:-1]
    curr_frames = images[1:]
    idx = 0
    for imfile1, imfile2 in zip(prev_frames, curr_frames):
        metrics = computeHS(p, imfile1, imfile2, 0.1, 0.1, path, idx)
        if metrics:
            all_metrics.append(metrics)
        idx += 1
        # if args.dev:
        #     if idx == 4:
        #         break
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

    plt.legend(handles=[line1, line2, line3, line4, line5])
    plt.xlabel('FPS')
    plt.ylabel('Error')
    plt.savefig(f'{path}/{cube_data}_horn_schunck_error.jpg')


# using multiprocessing
fps_metrics = {}
for fps in FPS:
    print(f'fps: {fps}')
    frame_path = f'{PATH}/{fps}/frame'
    os.makedirs(f'{PATH}/{fps}/frame/horn_schunck', exist_ok=True)
    all_metrics = horn_schunck(PATH, fps, frame_path)
    fps_metrics[fps] = all_metrics
    # with Pool(8) as p:
    #     p.starmap(horn_schunck, [(fps, path)])

plot_fps_vs_error(FPS, fps_metrics, PATH, DATA_FOLDER[args.data])
