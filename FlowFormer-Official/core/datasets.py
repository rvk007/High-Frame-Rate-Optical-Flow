# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import math
import os
import sys

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append('core')
import os.path as osp
import random
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        # img1 = frame_utils.read_gen(self.image_list[index][0])
        # img2 = frame_utils.read_gen(self.image_list[index][1])
        imgs = [frame_utils.read_gen(img) for img in self.image_list[index]]

        flow = np.array(flow).astype(np.float32)
        # img1 = np.array(img1).astype(np.uint8)
        # img2 = np.array(img2).astype(np.uint8)
        imgs = [np.array(img).astype(np.uint8) for img in imgs]

        # # grayscale images
        # if len(img1.shape) == 2:
        #     img1 = np.tile(img1[...,None], (1, 1, 3))
        #     img2 = np.tile(img2[...,None], (1, 1, 3))
        # else:
        #     img1 = img1[..., :3]
        #     img2 = img2[..., :3]

        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[...,None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]

        # if self.augmentor is not None:
        #     if self.sparse:
        #         img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
        #     else:
        #         img1, img2, flow = self.augmentor(img1, img2, flow)
        if self.augmentor is not None:
            if self.sparse:
                imgs, flow, valid = self.augmentor(imgs, flow, valid)
            else:
                imgs, flow = self.augmentor(imgs, flow)

        # img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        # img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        # print('Returning data imgs',len(imgs))
        return imgs, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean', fps=24):
        super(MpiSintel, self).__init__(aug_params)
        # flow_root = osp.join(root, split, 'flow')
        # image_root = osp.join(root, split, dstype)

        # if split == 'test':
        #     self.is_test = True

        # for scene in os.listdir(image_root):
        #     image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
        #     for i in range(len(image_list)-1):
        #         self.image_list += [ [image_list[i], image_list[i+1]] ]
        #         self.extra_info += [ (scene, i) ] # scene and frame_id

        #     if split != 'test':
        #         self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

        self.image_list, self.flow_list = [], []
        frame_path = f'/scratch/rr3937/optical_flow/FlowFormer-Official/datasets/{dstype}'
        flow_path = '/scratch/rr3937/optical_flow/FlowFormer-Official/datasets/flow'
        for scene in os.listdir(frame_path):
            flow = sorted(glob(osp.join(flow_path, scene, '*.flo')))
            images = sorted(glob(osp.join(frame_path, scene, '*.png')))
            for img1, img2, img3, img4 in zip(images[0::1], images[1::1], images[2::1], images[3::1]):
                self.image_list.append([img1.strip(), img2.strip(), img3.strip(), img4.strip()])
            for flo in flow[2::1]:
                self.flow_list.append(flo)

        # select frames based on FPS
        frame_freq = 24//fps
        self.image_list = self.image_list[::frame_freq]
        self.flow_list = self.flow_list[::frame_freq]

        assert (len(self.image_list) == len(self.flow_list))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass', split='training'):
        super(FlyingThings3D, self).__init__(aug_params)

        split_dir = 'TRAIN' if split == 'training' else 'TEST'
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'{split_dir}/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/{split_dir}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """
    print('Training on: ', TRAIN_DS)
    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean_fps_1 = MpiSintel(aug_params, split='training', dstype='clean', fps=1)
        sintel_clean_fps_4 = MpiSintel(aug_params, split='training', dstype='clean', fps=4)
        sintel_clean_fps_8 = MpiSintel(aug_params, split='training', dstype='clean', fps=8)
        sintel_clean_fps_24 = MpiSintel(aug_params, split='training', dstype='clean', fps=24)
        sintel_final_fps_1 = MpiSintel(aug_params, split='training', dstype='final', fps=1)
        sintel_final_fps_4 = MpiSintel(aug_params, split='training', dstype='final', fps=4)
        sintel_final_fps_8 = MpiSintel(aug_params, split='training', dstype='final', fps=8)
        sintel_final_fps_24 = MpiSintel(aug_params, split='training', dstype='final', fps=24)
        # sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        print('len(sintel_clean_fps_1): ', len(sintel_clean_fps_1))
        print('len(sintel_clean_fps_4): ', len(sintel_clean_fps_4))
        print('len(sintel_clean_fps_8): ', len(sintel_clean_fps_8))
        print('len(sintel_clean_fps_24): ', len(sintel_clean_fps_24))
        print('len(sintel_final_fps_1): ', len(sintel_final_fps_1))
        print('len(sintel_final_fps_4): ', len(sintel_final_fps_4))
        print('len(sintel_final_fps_8): ', len(sintel_final_fps_8))
        print('len(sintel_final_fps_24): ', len(sintel_final_fps_24))
        exit()


        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            # train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things
            train_dataset = 6*sintel_clean_fps_1 + 6*sintel_final_fps_24

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean_fps_24 + 100*sintel_final_fps_24 + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
        pin_memory=False, shuffle=True, num_workers=128, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
