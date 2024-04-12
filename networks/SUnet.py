import copy
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models import register_model

from timm.models.layers import to_2tuple, trunc_normal_
from networks.decoder import SUnetDecoder, Block
import timm


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class SUnetEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=9, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, dpr=None):
        super().__init__()
        cur = 0
        self.num_stages = num_stages

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward_features(self, x):
        B = x.shape[0]
        skips = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)

            x = norm(x)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i < self.num_stages - 1:
                skips.append(x)
        return x, skips

    def forward(self, x):
        x, skips = self.forward_features(x)
        return x, skips
class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

# self, img_size=224, patch_size=16, in_chans=3, num_classes=9, embed_dims=[64, 128, 320, 512],
#                  num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
#                  sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, dpr=None


    # def __init__(self, input_size=7, embed_dims_up=[768, 384, 192, 96], up_depths=[2, 2, 2], num_classes=9,
    #              num_heads=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4], qkv_bias=None, qk_scale=None, drop_rate=0.,
    #              attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
    #              sr_ratios=[8, 4, 2, 1], num_stage=4, linear=False, dpr=None):

class SUnet(nn.Module):
    def __init__(self, encoder_pretrained=True, patch_size=4, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],drop_path_rate=0.2, up_depths=[0, 2, 2, 2],
                 embed_dims_up=[512, 320, 128, 64], num_classes=9):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        self.encoder = SUnetEncoder(embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios,
                            qkv_bias=qkv_bias, norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios,
                            dpr=dpr)
        # print(self.encoder)
        self.decoder = SUnetDecoder(embed_dims_up=embed_dims_up, up_depths=up_depths, num_classes=num_classes,
                                 num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, sr_ratios=sr_ratios,
                                 dpr=dpr)
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=num_classes,
            upsampling=2
        )
        # print(self.decoder)
        # print(1)
        # self.sengmentation_heads =




    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x4, skips = self.encoder(x)
        out = self.decoder(x4, skips)
        out = self.segmentation_head(out)

        return out

    def load_from(self, pretrained_path):

        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            # if "model"  not in pretrained_dict:
            #     print("---start load pretrained modle by splitting---")
            #     pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            #     for k in list(pretrained_dict.keys()):
            #         if "output" in k:
            #             print("delete key:{}".format(k))
            #             del pretrained_dict[k]
            #     msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
            #     # print(msg)
            #     return
            # pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.encoder.state_dict()
            # full_dict = copy.deepcopy(pretrained_dict)
            # for k, v in pretrained_dict.items():
            #     if "block" in k:
            #         current_layer_num = 3-int(k[7:8])
            #         current_k = "layers_up." + str(current_layer_num) + k[8:]
            #         full_dict.update({current_k:v})
            # for k in list(full_dict.keys()):
            #     if k in model_dict:
            #         if full_dict[k].shape != model_dict[k].shape:
            #             print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
            #             del full_dict[k]

            msg = self.encoder.load_state_dict(pretrained_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")


if __name__ == "__main__":
    x = torch.rand((8, 1, 224, 224))

    net = PVTUnet()
    print(net)
    net.load_from('../pretrain_pth/pvt_v2_b1.pth')

    logits = net(x)
    print(logits)