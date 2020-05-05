from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn

from mmdet.core import auto_fp16, bbox2result, bbox_mask2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from IPython import embed
import time
import torch


@DETECTORS.register_module
class SiamPolarMask(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SiamPolarMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def extract_feat(self, img, img_refer):
        x = self.backbone(img, img_refer)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      img_refer,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      _gt_labels=None,
                      _gt_bboxes=None,
                      _gt_masks=None
                      ):

        if _gt_labels is not None:
            extra_data = dict(_gt_labels=_gt_labels,
                              _gt_bboxes=_gt_bboxes,
                              _gt_masks=_gt_masks)
        else:
            extra_data = None


        x = self.extract_feat(img, img_refer)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        
        losses = self.bbox_head.loss(
            *loss_inputs,
            gt_masks = gt_masks,
            gt_bboxes_ignore=gt_bboxes_ignore,
            extra_data=extra_data
        )
        return losses

    def forward_test(self, imgs, img_metas, img_refers, rescale):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas'), (img_refers, 'img_refers')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], img_refers[0], rescale)
        else:
            return self.aug_test(imgs, img_metas, img_refers, rescale)

    def simple_test(self, img, img_meta, img_refer, rescale=False):
        x = self.extract_feat(img, img_refer)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        results = [
            bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_meta[0])
            for det_bboxes, det_labels, det_masks in bbox_list]

        bbox_results = results[0][0]
        mask_results = results[0][1]

        return bbox_results, mask_results

    @auto_fp16(apply_to=('img', ))
    @auto_fp16(apply_to=('img_refer', ))
    def forward(self, img, img_meta, img_refer, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, img_refer, **kwargs)
        else:
            return self.forward_test(img, img_meta, img_refer, **kwargs)
