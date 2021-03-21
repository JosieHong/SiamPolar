import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BACKBONES
from mmcv.cnn import constant_init, kaiming_init
from .resnet import ResNet

class GCN(nn.Module):
    def __init__(self, c, out_c, k=7): # out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k,1), padding=((k-1)//2,0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1,k), padding=(0,(k-1)//2))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k), padding=((k-1)//2,0))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k,1), padding=(0,(k-1)//2))
        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        
        x = x_l + x_r
        
        return x

@BACKBONES.register_module
class SiamResNetGCN(nn.Module):
    """ This is a simese network using ResNet bacbone and returning every 
    blocks' feature map.
    """
    def __init__(self, 
                 depth,
                 template_depth,
                 template_pretrained,
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
                 zero_init_residual=True,
                 correlation_blocks=[3, 4, 5],
                 attention_blocks=None):
        super(SiamResNetGCN, self).__init__()
        self.template_backbone = ResNet(template_depth, 
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
        self.template_pretrained = template_pretrained
        self.search_backbone = ResNet(depth,
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
        # Cross Correlation
        self.correlation_blocks = [correlation_block-2 
                                    for correlation_block in correlation_blocks] # start from block2
        self.match_batchnorm = nn.BatchNorm2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.gama = nn.Parameter(torch.zeros(1))

        # GCNs
        self.gcn = nn.ModuleList([GCN(256, 256, k=7), GCN(512, 512, k=7), GCN(1024, 1024, k=7), GCN(2048, 2048, k=7)])

    def forward(self, x1, x2): 
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

        # extract features
        search_blocks = self.search_backbone(x1)
        template_blocks = self.template_backbone(x2)
        # init outs
        outs = [self.gcn[i](search_block) for i, search_block in enumerate(search_blocks)]

        # re-cross correlation
        for correlation_block in self.correlation_blocks:
            embedding_search = search_blocks[correlation_block]
            embedding_template = template_blocks[correlation_block]

            # re-correlation
            match_map = self.match_corr(embedding_search, 
                                        embedding_template, 
                                        embedding_search.shape[2:])
            match_map = match_map.repeat(1, embedding_template.size()[1], 1, 1)
            corr_value = self.softmax(match_map)*embedding_search
            outs[correlation_block] =  self.gama*corr_value + embedding_search
        
        return tuple(outs)
    
    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        self.search_backbone.init_weights(pretrained)
        self.template_backbone.init_weights(self.template_pretrained)
    
    def match_corr(self, embed_srch, embed_ref, upsc_size):
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
        # print('embed_srch: ', embed_srch.size(), "embed_ref: ", embed_ref.size())
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