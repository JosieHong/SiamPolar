'''
@Author: JosieHong
@Date: 2020-04-22 16:26:29
@LastEditAuthor: JosieHong
@LastEditTime: 2020-05-10 17:38:44
'''

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmdet.models.backbones.siamese_res import ResNet, Bottleneck, BasicBlock

@BACKBONES.register_module
class SiamBackbone(nn.Module):
    """ This is a simese network using resnet bacbone and returning every 
    blocks' feature map.
    """

    def __init__(self, 
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        super(SiamBackbone, self).__init__()
        self.resnet = ResNet(depth,
                            num_stages,
                            strides,
                            dilations,
                            out_indices,
                            style,
                            frozen_stages,
                            conv_cfg,
                            norm_cfg,
                            norm_eval,
                            dcn,
                            stage_with_dcn,
                            gcb,
                            stage_with_gcb,
                            gen_attention,
                            stage_with_gen_attention,
                            with_cp,
                            zero_init_residual)
        self.match_batchnorm = nn.BatchNorm2d(1)
        self.gen_block5 = nn.Conv2d(2049, 2048, 1)

    # Unfinished: 
    def forward(self, x1, x2):  # should return a tuple
        """
        Args:
            x1 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [4, 3, 255, 255].
            x2 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [4, 3, 127, 127].
        Returns:
            block2, block3, block4, block5 (embedding_search + match_map) 
                (torch.Tensor): Usually the shape is [].
        """
        # josie.debug
        # print(type(x1), type(x2))
        # print("x1 shape: {}, x2 shape: {}".format(x1.shape, x2.shape))
        # exit()
        
        # [4,256,64,64], [4,512,32,32], [4,1024,16,16], [4,2048,8,8]
        block2, block3, block4, embedding_search = self.resnet(x1)
        # [4,2048,4,4]
        _, _, _, embedding_reference = self.resnet(x2)

        # josie.debug
        # print("ResNet out shape: {}, {}, {}, {}, {}".format(embedding_reference.shape, block2.shape, block3.shape, \
        #     block4.shape, embedding_search.shape))
        # exit()

        self.upsc_size = embedding_search.shape[2:] # [8,8]
        # [4,1,5,5] -> [4,1,8,8]
        match_map = self.match_corr(embedding_reference, embedding_search, self.upsc_size)

        # josie.debug
        # print("match_map shape: {}".format(match_map.shape))
        # exit()

        block5 = self.gen_block5(torch.cat((embedding_search, match_map), dim=1))

        # josie.debug
        # print("block5 shape: {}".format(block5.shape))
        # exit()
        
        # [4,256,64,64], [4,512,32,32], [4,1024,16,16], [4,2048,8,8]
        out = [block2, block3, block4, block5]

        return tuple(out)
    
    def init_weights(self, pretrained=None):
        self.resnet.init_weights(pretrained)

    def match_corr(self, embed_ref, embed_srch, upsc_size):
        """ reference: https://github.com/rafellerc/Pytorch-SiamFC
        Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].
        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.
        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b)
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)

        match_map = F.interpolate(match_map, upsc_size, mode='bilinear', align_corners=False)

        return match_map