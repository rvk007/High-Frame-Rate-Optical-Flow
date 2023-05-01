import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum
from utils.utils import bilinear_sampler, coords_grid, upflow8

from ...position_encoding import LinearPositionEncoding, PositionEncodingSine
from ..common import (
    MLP,
    FeedForward,
    MultiHeadAttention,
    pyramid_retrieve_tokens,
    retrieve_tokens,
    sampler,
    sampler_gaussian_fix,
)
from ..encoders import twins_svt_large, twins_svt_large_context
from .cnn import BasicEncoder
from .decoder import MemoryDecoder
from .encoder import MemoryEncoder
from .twins import PosConv


class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain, use_prior=self.cfg.use_prior, cnet=True)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, imgs, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        imgs = [2 * (img / 255.0) - 1.0 for img in imgs]
        image1, image2 = imgs[-2], imgs[-1]

        data = {}
        # FIXME: Using previous frames as context
        self.cfg.context_concat = True
        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat(imgs, dim=1), use_prior=self.cfg.use_prior)
        else:
            context = self.context_encoder(image1)
        # print('In FlowFormer context',context.shape)

        cost_memory = self.memory_encoder(image1, image2, data, context)
        # print('In FlowFormer cost_memory',cost_memory.shape)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)
        # print('In FlowFormer flow_predictions',len(flow_predictions), flow_predictions[0].shape)

        return flow_predictions
