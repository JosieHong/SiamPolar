'''
@Author: JosieHong
@Date: 2020-06-06 21:51:27
@LastEditAuthor: JosieHong
LastEditTime: 2021-06-24 15:58:36
'''
import torch.nn as nn
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule

@NECKS.register_module
class Single_Connect(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 out_blocks=[5],
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(Single_Connect, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_blocks, list)
        self.index = [i-2 for i in out_blocks] # convert block number to index

        self.lateral_convs = nn.ModuleList()
        for i in range(len(out_blocks)):
            l_conv = ConvModule(
                    in_channels[self.index[i]],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=activation,
                    inplace=False)
            self.lateral_convs.append(l_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        outs = [
            lateral_conv(inputs[self.index[i]])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        return tuple(outs)
