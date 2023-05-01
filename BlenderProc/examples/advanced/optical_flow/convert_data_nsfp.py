import os
import h5py
import numpy as np

PATH = '/Users/rakhee/Projects/ComputerVision/optical_flow/kitti_updated/'
data_path = '/scratch/rr3937/optical_flow/BlenderProc/examples/advanced/optical_flow/output/cube_asphalt'
FPS = [1, 4, 8, 16, 24, 30, 60]
FPS = [1]
for fps in FPS:
    print(fps)
    os.makedirs(f'{data_path}/{fps}/ground_truth_flow', exist_ok=True)
    os.makedirs(f'{data_path}/{fps}/frame', exist_ok=True)
    for i in range(10 * fps):
        with h5py.File(f'{PATH}/{fps}/{i}.hdf5', 'r') as data:
            flow = data['forward_flow'][:]
            data = data['colors'][:]


for f in os.listdir(PATH):
    print(f)
    data = np.load(PATH + f)
    pc1 = data['pos1'][:]
    pc2 = data['pos2'][:]
    gt = data['gt'][:]
    pc1[:,-1] = 0.0
    pc2[:,-1] = 0.0
    gt[:,-1] = 0.0
    np.savez_compressed('/Users/rakhee/Projects/ComputerVision/optical_flow/kitti_updated/' + f, pos1=pc1, pos2=pc2, gt=gt)
