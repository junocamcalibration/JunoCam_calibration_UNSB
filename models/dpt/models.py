import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    FeatureFusionBlock_UNSB,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

def _make_fusion_block_UNSB(features):
    return FeatureFusionBlock_UNSB(features)


class DPT(BaseModel):
    def __init__(
        self,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            use_pretrained=True, #False,  # Set to true if you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block_UNSB(features)
        self.scratch.refinenet2 = _make_fusion_block_UNSB(features)
        self.scratch.refinenet3 = _make_fusion_block_UNSB(features)
        self.scratch.refinenet4 = _make_fusion_block_UNSB(features)


    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        
        # These layers change the number of channels to 256 from the vit backbone
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # These layer fuse the multiple scales
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        '''
        layer1: torch.Size([1, 256, 64, 64])
        layer2: torch.Size([1, 512, 32, 32])
        layer3: torch.Size([1, 768, 16, 16])
        layer4: torch.Size([1, 768, 8, 8])
        layer1_rn: torch.Size([1, 256, 64, 64])
        layer2_rn: torch.Size([1, 256, 32, 32])
        layer3_rn: torch.Size([1, 256, 16, 16])
        layer4_rn: torch.Size([1, 256, 8, 8])
        path_4: torch.Size([1, 256, 16, 16])
        path_3: torch.Size([1, 256, 32, 32])
        path_2: torch.Size([1, 256, 64, 64])
        path_1: torch.Size([1, 256, 128, 128])
        '''

        return path_1, path_2
