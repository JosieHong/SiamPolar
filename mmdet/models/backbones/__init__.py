'''
@Author: open-mmlab
@Date: 2020-04-22 15:08:46
@LastEditAuthor: JosieHong
@LastEditTime: 2020-06-16 10:29:12
'''

from .hrnet import HRNet
from .resnet import ResNet, make_res_layer
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
# from .siamese import SiamResNet, SiamResNeXt, AsySiamNet
from .siam_resnet import SiamResNet
from .siam_resnet_gcn import SiamResNetGCN

__all__ = ['ResNet', 'make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet', 
            'SiamResNet', 'SiamResNetGCN']
