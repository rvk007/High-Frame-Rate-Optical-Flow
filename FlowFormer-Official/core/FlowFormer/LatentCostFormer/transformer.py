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
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(input_dim=12, output_dim=256, norm_fn='instance')


    def forward(self, imgs, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        imgs = [2 * (img / 255.0) - 1.0 for img in imgs]
        image1, image2 = imgs[-2], imgs[-1]

        data = {}
        # FIXME: Using previous frames as context
        self.cfg.context_concat = True
        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat(imgs, dim=1))
        else:
            context = self.context_encoder(image1)
        print('In FlowFormer context',context.shape) # context [6, 256, 54, 120]

        cost_memory = self.memory_encoder(image1, image2, data, context)
        print('In FlowFormer cost_memory',cost_memory.shape) # cost_memory [19440, 8, 128]

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)
        print('In FlowFormer flow_predictions',len(flow_predictions), flow_predictions[0].shape) # flow_predictions [2, 2, 432, 960]

        return flow_predictions

# There is no conclusion from these experiments. The results are not stable. So, I'll run experiments with them and see what happens.
# With BasicEncoder as context_encoder
# self.running_loss:  {
#     'epe': 1327.9566848278046,
#     '1px': 32.65624411776662,
#     '3px': 53.433986853808165,
#     '5px': 62.59806661307812,
#     '5-th-5px': 74.25049888552167,
#     '10-th-5px': 71.41045503690839,
#     '20-th-5px': 69.39342239499092
#     }
# self.running_loss:  {'epe': 1226.9535936117172,
# '1px': 42.724916107952595,
# '3px': 57.96929378807545,
# '5px': 65.18931993842125,
# '5-th-5px': 76.68895510211587,
# '10-th-5px': 75.02072904724628,
# '20-th-5px': 71.50594470277429
# }

# Modidy the svt patch_embedding to 12 channels
# self.running_loss:  {'epe': 1092.3259217143059,
# '1px': 52.361195757985115,
# '3px': 65.01739060878754,
# '5px': 71.38004159927368,
# '5-th-5px': 81.56093208305538,
# '10-th-5px': 79.50140569731593,
# '20-th-5px': 77.37747713923454
# }
# self.running_loss:  {'epe': 1127.045578122139,
# '1px': 50.39959220588207,
# '3px': 62.491314113140106,
# '5px': 68.95085415244102,
# '5-th-5px': 78.29116517677903,
# '10-th-5px': 76.92328018415719,
# '20-th-5px': 74.19069255515933
# }

# Original with no concat
# self.running_loss:  {'epe': 959.0277824401855
# '1px': 55.620319589972496
# '3px': 67.74122357368469
# '5px': 73.79586663842201
# '5-th-5px': 82.97825861582533
# '10-th-5px': 81.28914101421833
# '20-th-5px': 78.8425826728344
# }
# self.running_loss:  {'epe': 1032.4617593288422,
# '1px': 52.54213923215866,
# '3px': 64.57677978277206,
# '5px': 70.84950187802315,
# '5-th-5px': 79.51233647670597,
# '10-th-5px': 78.45759001374245,
# '20-th-5px': 75.67630953062326
# }

# Experiment to run
# 1. Use BasicEncoder as context_encoder
# 2. Modidy the svt patch_embedding to 12 channels