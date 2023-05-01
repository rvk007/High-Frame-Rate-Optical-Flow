import os

import cv2
import matplotlib
import numpy as np

matplotlib.use('pdf')
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--data')
args = parser.parse_args()

TIME_IN_SECONDS = 10
ALGORITHMS = ['lucas_kanade', 'lucaskanade_dense', 'horn_schunck', 'farneback', 'rlof']
PATH = '/scratch/rr3937/optical_flow/BlenderProc/examples/advanced/optical_flow/output'
cube_data  = {1: 'cube_asphalt', 2:'cube_brick', 3:'cube_pavement'}
data = cube_data[int(args.data)]

if args.dev:
    FPS = [1]
else:
    FPS = [16] #, 1, 4, 8,,24, 30, 60

# plt.figure(figsize = (8,1))
plt.rcParams['font.size'] = 5
gs1 = gridspec.GridSpec(1,8)
gs1.update(wspace=0.35, hspace=0.25) # set the spacing between axes.

print(f'data: {data}')
for fps in FPS:
    print(fps)
    for idx in range(TIME_IN_SECONDS*fps):
        if os.path.exists(f'{data}_ocv/{fps}/{idx}.jpg'):
            continue
        else:
            x = [cv2.imread(f'{PATH}/{data}/{fps}/frame/{idx}.png')]
            for al in ALGORITHMS:
                file_path = f'{PATH}/{data}/{fps}/frame/{al}/{idx}.jpg'
                if os.path.exists(file_path):
                    x.append(cv2.imread(file_path))
                else:
                    x.append(np.zeros((1024, 1024, 3), dtype=np.uint8))
            os.makedirs(f'{data}_ocv/{fps}', exist_ok=True)
            ax = plt.subplot(gs1[0])
            ax.imshow(x[0])
            ax.set_title('frame')
            ax.axis('off')
            ax.set_aspect('equal')
            for i_al, al in enumerate(ALGORITHMS):
                ax = plt.subplot(gs1[i_al+1])
                ax.imshow(x[i_al+1])
                if al == 'lucaskanade_dense':
                    ax.set_title('lk_dense')
                else:
                    ax.set_title(al)
                ax.set_aspect('equal')
                ax.axis('off')
            plt.savefig(f'{data}_ocv/{fps}/{idx}.jpg', dpi=480, bbox_inches='tight', pad_inches=0.1)

# import numpy as np
# import matplotlib.pyplot as plt

# # Create numpy array
# arr = np.random.rand(4, 10)

# # Create figure and subplots
# fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

# # Plot subplots
# for i in range(4):
#     axs[i].plot(arr[i])
#     axs[i].set_title("Subplot {}".format(i+1))

# # Set figure title and show plot
# fig.suptitle("Four subplots of a numpy array", fontsize=14)
# plt.savefig('test.jpg', dpi=480, bbox_inches='tight', pad_inches=0.1)
