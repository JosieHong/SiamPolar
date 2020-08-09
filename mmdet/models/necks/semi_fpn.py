'''
@Author: JosieHong
@Date: 2020-06-06 21:51:27
@LastEditAuthor: JosieHong
@LastEditTime: 2020-07-31 12:26:07
'''
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class SemiFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(SemiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
    
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # inputs:
        # torch.Size([8, 256, 64, 64])
        # torch.Size([8, 512, 32, 32])
        # torch.Size([8, 1024, 16, 16])
        # torch.Size([8, 2048, 8, 8])
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels-1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i-1].size()[2:], mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(orig)
                else:
                    outs.append(outs[-1])
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(F.relu(outs[-1]))
                    else:
                        outs.append(outs[-1])
        
        # num_outs = 3
        # torch.Size([8, 256, 32, 32])
        # torch.Size([8, 256, 16, 16])
        # torch.Size([8, 256, 8, 8])

        # num_outs = 4
        # torch.Size([8, 256, 32, 32])
        # torch.Size([8, 256, 16, 16])
        # torch.Size([8, 256, 8, 8])
        # torch.Size([8, 256, 4, 4])

        # # part 3: from bottom to top
        # reverses = [
        #     F.avg_pool2d(outs[i], kernel_size=1, stride=2) for i in range(len(outs)-1)
        # ]
        # for i in range(1, self.num_outs):
        #     outs[i] = outs[i] + reverses[i-1]
        
        # for out in outs:
        #     print(out.size())
        # exit()
        return tuple(outs)

