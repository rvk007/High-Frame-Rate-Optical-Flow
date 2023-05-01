import argparse
import os
from multiprocessing import Pool

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', default="examples/advanced/optical_flow/output/cube", help="Path to where the final files, will be saved")
args = parser.parse_args()
PATH = args.path

TIME_IN_SECONDS = 10
FPS = [60]
# FPS = [4, 8, 16, 24, 30, 60]
# FPS = [30, 60]

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

def save_frame_gt(fps, i):
    with h5py.File(f'{PATH}/{fps}/{i}.hdf5', 'r') as data:
        forward_flow = flow_to_rgb(data['forward_flow'][:])
        frame = data['colors'][:]
        plt.imsave(f'{PATH}/{fps}/ground_truth_flow/{i}.png', forward_flow)
        plt.imsave(f'{PATH}/{fps}/frame/{i}.png', frame)

# use multiprocessing to speed up the process
for fps in FPS:
    print(fps)
    os.makedirs(f'{args.path}/{fps}/ground_truth_flow', exist_ok=True)
    os.makedirs(f'{args.path}/{fps}/frame', exist_ok=True)
    with Pool(8) as p:
        p.starmap(save_frame_gt, [(fps, i) for i in range(TIME_IN_SECONDS * fps)])


# idxs = [0,1]
# with h5py.File(f'{PATH}/4/0.hdf5', 'r') as data:
#     flow_4_0 = flow_to_rgb(data['forward_flow'][:])

# with h5py.File(f'{PATH}/4/1.hdf5', 'r') as data:
#     flow_4_1 = flow_to_rgb(data['forward_flow'][:])

# with h5py.File(f'{PATH}/4/2.hdf5', 'r') as data:
#     flow_4_2 = flow_to_rgb(data['forward_flow'][:])

# with h5py.File(f'{PATH}/4/5.hdf5', 'r') as data:
#     flow_4_5 = flow_to_rgb(data['forward_flow'][:])

# for i in range(1, 10):
#     with h5py.File(f'{PATH}/16/{i}.hdf5', 'r') as data:
#         flow_16 = flow_to_rgb(data['forward_flow'][:])
#     for j in range(10):
#         with h5py.File(f'{PATH}/8/{j}.hdf5', 'r') as data:
#             flow_8 = flow_to_rgb(data['forward_flow'][:])
#         if np.mean(np.abs(flow_16 - flow_8))< 0.001:
#             print(f'{i}: {j}')

# with h5py.File(f'{PATH}/8/0.hdf5', 'r') as data:
#     flow_8_0 = flow_to_rgb(data['forward_flow'][:])

# with h5py.File(f'{PATH}/8/1.hdf5', 'r') as data:
#     flow_8_1 = flow_to_rgb(data['forward_flow'][:])

# with h5py.File(f'{PATH}/8/2.hdf5', 'r') as data:
#     flow_8_2 = flow_to_rgb(data['forward_flow'][:])

# with h5py.File(f'{PATH}/8/3.hdf5', 'r') as data:
#     flow_8_3 = flow_to_rgb(data['forward_flow'][:])


# with h5py.File(f'{PATH}/8/79.hdf5', 'r') as data:
#     flow_8_79 = flow_to_rgb(data['forward_flow'][:])

# with h5py.File(f'{PATH}/4/39.hdf5', 'r') as data:
#     flow_4_39 = flow_to_rgb(data['forward_flow'][:])

# print('For 0', np.mean(np.abs(flow_4_0 - flow_8_0)))
# print('For 2', np.mean(np.abs(flow_4_0 - flow_8_1)))
# print('For 1', np.mean(np.abs(flow_4_0 - flow_8_2)))

# print()
# print('For 0', np.mean(np.abs(flow_4_1 - flow_8_0)))
# print('For 2', np.mean(np.abs(flow_4_1 - flow_8_1)))
# print('For 1', np.mean(np.abs(flow_4_1 - flow_8_2)))
# print('For 1', np.mean(np.abs(flow_4_1 - flow_8_3)))

# print('Final', np.mean(np.abs(flow_4_39 - flow_8_79)))