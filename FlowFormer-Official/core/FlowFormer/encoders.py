import numpy as np
import timm
import torch
import torch.nn as nn

from .LatentCostFormer.cnn import BasicEncoder

# from transformers import GPT2Config, GPT2Model



class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=pretrained)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]

        # self.transformers_config = GPT2Config(
        #     vocab_size = 0,
        #     n_positions = 4,
        #     n_embd = 6480, #1658880, #512,
        #     n_layer = 2,
        #     n_head = 4
        # )
        # `embed_dim` must be divisible by num_heads

    def forward(self, x, data=None, layer=2):
        B, C, _, _ = x.shape
        print('In twins_svt_large Input shape', x.shape)
        # 2, 12, 432, 960
        # FIXME: Find a better a day to do this; Right now works for 4 frames given as input
        if C > 3:
            print('self.svt.patch_embeds[0]', self.svt.patch_embeds[0])
            self.svt.patch_embeds[0].proj = nn.Conv2d(12, 128, kernel_size=(4, 4), stride=(4, 4)).to('cuda')
            print('self.svt.patch_embeds[0]', self.svt.patch_embeds[0])
        #     # FIXME: This needs to be changed with a GPT or other feature extractor for the context from previous frames
        #     # input_dim = C
        #     # self.conv1 = nn.Conv2d(input_dim, 9, kernel_size=1).cuda()
        #     # self.relu1 = nn.ReLU(inplace=True).cuda()
        #     # self.conv2 = nn.Conv2d(9, 3, kernel_size=1).cuda()
        #     # self.relu2 = nn.ReLU(inplace=True).cuda()
        #     # x = self.relu1(self.conv1(x))
        #     # x = self.relu2(self.conv2(x))
        #     # print('In twins_svt_large x', x.shape)

        #     prev_context = BasicEncoder(input_dim=C, output_dim=256, norm_fn='instance').to('cuda')
        #     context = prev_context(x)
        #     # print('In twins_svt_large prev_context', context.shape)
        #     return context

        #     embeddings = x
        #     model_previous_frames = GPT2Model(self.transformers_config).cuda()
        #     logits = model_previous_frames(
        #         inputs_embeds=embeddings
        #     )
        #     print('logits', logits.shape)
        #     logits = logits[:, -1, :]
        #     print('In twins_svt_large logits', logits.shape)
        #     x = logits

        # The output of the transformer model should be of channel 3 and weight is [128, 3, 4, 4]
        # print('length of layers in twins_svt_large', len(self.svt.patch_embeds), len(self.svt.pos_drops), len(self.svt.blocks), len(self.svt.pos_block))
        # 2 4 2 2
        # data = x
        # a = []
        # for r in range(0, 12, 3):
        #     print('range', r)
        #     x = data[:, r:r+3, :, :]
        #     print('x', x.shape)

        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            # print('x before embed', x.shape)
            x, size = embed(x)
            # print('In twins_svt_large x', x.shape, ' size', size)
            x = drop(x)
            # print('In twins_svt_large drop x', x.shape)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            # print('self.svt.depths', len(self.svt.depths))
            if i < len(self.svt.depths) - 1:
                # print('before reshape x', x.shape)
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
                # print('after reshape x', x.shape)

            if i == layer-1:
                break
            # x = x.unsqueeze(1)
            # a.append(x)

        # print(len(a))
        # x = torch.cat(a, dim=1)
        # print('Done In twins_svt_large x', x.shape)
        # x = x.reshape(B, 4, -1)
        # print('After reshape In twins_svt_large x', x.shape)
        # 2, 256, 54, 120
        # 2 4 256 54 120 -> take the last one -> 2 1 256 54 120 -> 2 256 54 120
        return x

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            for param in embed.parameters():
                num +=  np.prod(param.size())

            for param in drop.parameters():
                num +=  np.prod(param.size())

            for param in blocks.parameters():
                num +=  np.prod(param.size())

            for param in pos_blk.parameters():
                num +=  np.prod(param.size())

            if i == layer-1:
                break

        for param in self.svt.head.parameters():
            num +=  np.prod(param.size())

        return num

class twins_svt_large_context(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large_context', pretrained=pretrained)

    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer-1:
                break

        return x


if __name__ == "__main__":
    m = twins_svt_large()
    input = torch.randn(2, 3, 400, 800)
    out = m.extract_feature(input)
    print(out.shape)
