import blenderproc as bproc
import argparse
import os


import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('camera', nargs='?', default="examples/advanced/optical_flow/camera_positions", help="Path to the camera file")
parser.add_argument('scene', nargs='?', default="examples/resources/scene.obj", help="Path to the scene.obj file")
parser.add_argument('output_dir', nargs='?', default="examples/advanced/optical_flow/output", help="Path to where the final files, will be saved")
parser.add_argument('part', nargs='?')
args = parser.parse_args()

bproc.init()
TIME_IN_SECONDS = 10
# FPS = [60]
# FPS = [4, 8, 16, 24, 30, 60]
FPS = []
print("Part: ", args.part)
if int(args.part) ==  1:
    FPS = [4, 8, 16]
if int(args.part) ==  2:
    FPS = [24, 30]
if int(args.part) ==  3:
    FPS = [60]

# load the objects into the scene
for fps in FPS:
    print("FPS: ", fps)
    objs = bproc.loader.load_obj(args.scene)
    print("Number of objects in the scene: ", len(objs))
    print("Obj locations: ", [obj.get_location() for obj in objs])
    pos_t = np.linspace(4, -1, num=TIME_IN_SECONDS * fps, endpoint=True, retstep=False, dtype=None, axis=0)
    pos_s = np.linspace(1, 6.2, num=TIME_IN_SECONDS * fps, endpoint=True, retstep=False, dtype=None, axis=0)
    pos_tl = np.linspace(0, 4, num=TIME_IN_SECONDS * fps, endpoint=True, retstep=False, dtype=None, axis=0)
    pos_j = np.linspace(-2, 4.2, num=TIME_IN_SECONDS * fps, endpoint=True, retstep=False, dtype=None, axis=0)

    # print("Total number of frames: ",pos.shape)
    for i in range(TIME_IN_SECONDS * fps):
        objs[0].set_location([0,0,pos_t[i]],i+1) # triangle
        objs[1].set_location([0,pos_s[i],0],i+1) # jellyfish
        objs[2].set_location([0,0,pos_tl[i]],i+1) # bear
        objs[3].set_location([pos_j[i],0,0],i+1) # tail

    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    # define the camera intrinsics
    bproc.camera.set_resolution(1024, 1024)

    # read the camera positions file and convert into homogeneous camera-world transformation
    with open(args.camera, "r") as f:
        for line in f.readlines():
            line = [float(x) for x in line.split()]
            position, euler_rotation = line[:3], line[3:6]
            matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)

    for _ in range(TIME_IN_SECONDS * fps):
        bproc.camera.add_camera_pose(matrix_world)

    # render the whole pipeline
    data = bproc.renderer.render()

    # Render the optical flow (forward and backward) for all frames
    data.update(bproc.renderer.render_optical_flow(get_backward_flow=True, get_forward_flow=True, blender_image_coordinate_style=False))

    # write the data to a .hdf5 container
    scene = args.scene.split("/")[-1].split(".")[0]
    os.makedirs(args.output_dir+f'/{scene}/{fps}', exist_ok=True)
    bproc.writer.write_hdf5(args.output_dir+f'/{scene}/{fps}', data)
    bproc.clean_up()
