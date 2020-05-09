'''
@Author: JosieHong
@Date: 2020-04-26 12:40:11
@LastEditAuthor: JosieHong
@LastEditTime: 2020-05-06 13:27:50
'''


import os.path as osp
import warnings

import mmcv
import numpy as np
from imagecorruptions import corrupt
from mmcv.parallel import DataContainer as DC
import torch

from .utils import random_scale, to_tensor
from .registry import DATASETS
from .coco_seg import Coco_Seg_Dataset, INF


@DATASETS.register_module
class DAVIS_Seg_Dataset(Coco_Seg_Dataset):

    CLASSES = ('aerobatics', 'bear', 'bike-packing', 'blackswan', 'bmx-bumps', 
                'bmx-trees', 'boat', 'boxing-fisheye', 'breakdance', 'breakdance-flare', 
                'bus', 'camel', 'car-race', 'car-roundabout', 'car-shadow', 
                'car-turn', 'carousel', 'cat-girl', 'cats-car', 'chamaleon', 
                'classic-car', 'color-run', 'cows', 'crossing', 'dance-jump', 
                'dance-twirl', 'dancing', 'deer', 'disc-jockey', 'dog', 
                'dog-agility', 'dog-gooses', 'dogs-jump', 'dogs-scale', 'drift-chicane', 
                'drift-straight', 'drift-turn', 'drone', 'elephant', 'flamingo',
                'giant-slalom', 'girl-dog', 'goat', 'gold-fish', 'golf',
                'guitar-violin', 'gym', 'helicopter', 'hike', 'hockey', 
                'horsejump-high', 'horsejump-low', 'horsejump-stick', 'hoverboard', 'india',
                'judo', 'kid-football', 'kite-surf', 'kite-walk', 'koala',
                'lab-coat', 'lady-running', 'libby', 'lindy-hop', 'loading', 
                'lock', 'longboard', 'lucia', 'mallard-fly', 'mallard-water', 
                'man-bike', 'mbike-trick', 'miami-surf', 'monkeys-trees', 'motocross-bumps', 
                'motocross-jump', 'motorbike', 'mtb-race', 'night-race', 'orchid', 
                'paragliding', 'paragliding-launch', 'parkour', 'people-sunset', 'pigs',
                'planes-crossing', 'planes-water', 'rallye', 'rhino', 'rollerblade', 
                'rollercoaster', 'salsa', 'schoolgirls', 'scooter-black', 'scooter-board',
                'scooter-gray', 'seasnake', 'sheep', 'shooting', 'skate-jump', 
                'skate-park', 'slackline', 'snowboard', 'soapbox', 'soccerball', 
                'stroller', 'stunt', 'subway', 'surf', 'swing', 
                'tandem', 'tennis', 'tennis-vest', 'tractor', 'tractor-sand', 
                'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking')
                
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 refer_scale=(127,127), 
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 corruption=None,
                 corruption_severity=1,
                 skip_img_without_anno=True,
                 test_mode=False):
        super(DAVIS_Seg_Dataset, self).__init__(ann_file,
                                                img_prefix,
                                                img_scale,
                                                img_norm_cfg,
                                                multiscale_mode,
                                                size_divisor,
                                                proposal_file,
                                                num_max_proposals,
                                                flip_ratio,
                                                with_mask,
                                                with_crowd,
                                                with_label,
                                                with_semantic_seg,
                                                seg_prefix,
                                                seg_scale_factor,
                                                extra_aug,
                                                resize_keep_ratio,
                                                corruption,
                                                corruption_severity,
                                                skip_img_without_anno,
                                                test_mode)
        self.refer_scale = refer_scale

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # corruption
        if self.corruption is not None:
            img = corrupt(
                img,
                severity=self.corruption_severity,
                corruption_name=self.corruption)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)

        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(img, img_scale, flip, keep_ratio=self.resize_keep_ratio)

        img = img.copy()

        # get img_refer from first frame
        first_frame_idx = img_info["first_frame"]
        refer_info = self.img_infos[first_frame_idx]
        refer_ann = self.get_ann_info(first_frame_idx)
        img_refer = mmcv.imread(osp.join(self.img_prefix, refer_info['filename']))
        # crop the bbox
        img_refer = torch.squeeze(torch.Tensor(mmcv.imcrop(img_refer, refer_ann["bboxes"])))
        # resize to refer_scale
        img_refer = torch.Tensor(mmcv.imresize(np.float32(img_refer), self.refer_scale, return_scale=False)).permute(2, 0, 1)

        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix,
                         img_info['filename'].replace('jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack([proposals, scores
                                   ]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            img_refer=DC(to_tensor(img_refer), stack=True))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        #--------------------offline ray label generation-----------------------------
        self.center_sample = True
        self.use_mask_center = True
        self.radius = 1.5
        self.strides = [8, 16, 32, 64, 128]
        self.regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))
        featmap_sizes = self.get_featmap_size(pad_shape)
        self.featmap_sizes = featmap_sizes
        num_levels = len(self.strides)
        all_level_points = self.get_points(featmap_sizes)
        self.num_points_per_level = [i.size()[0] for i in all_level_points]

        expanded_regress_ranges = [
            all_level_points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                all_level_points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, 0)
        gt_masks = gt_masks[:len(gt_bboxes)]

        gt_bboxes = torch.Tensor(gt_bboxes)
        gt_labels = torch.Tensor(gt_labels)

        _labels, _bbox_targets, _mask_targets = self.polar_target_single(
            gt_bboxes,gt_masks,gt_labels,concat_points, concat_regress_ranges)

        data['_gt_labels'] = DC(_labels)
        data['_gt_bboxes'] = DC(_bbox_targets)
        data['_gt_masks'] = DC(_mask_targets)
        #--------------------offline ray label generation-----------------------------
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # corruption
        if self.corruption is not None:
            img = corrupt(
                img,
                severity=self.corruption_severity,
                corruption_name=self.corruption)
        # load proposals if necessary
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None
        
        # get img_refer from first frame
        first_frame_idx = img_info["first_frame"]
        refer_info = self.img_infos[first_frame_idx]
        refer_ann = self.get_ann_info(first_frame_idx)
        img_refer = mmcv.imread(osp.join(self.img_prefix, refer_info['filename']))
        # crop the bbox
        img_refer = torch.squeeze(torch.Tensor(mmcv.imcrop(img_refer, refer_ann["bboxes"])))
        # resize to refer_scale
        img_refer = torch.Tensor(mmcv.imresize(np.float32(img_refer), self.refer_scale, return_scale=False)).permute(2, 0, 1)

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack([_proposal, score
                                       ]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        img_refers = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            img_refers.append(DC(to_tensor(img_refer), stack=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                img_refers.append(DC(to_tensor(img_refer), stack=True))
                proposals.append(_proposal)
        data = dict(img=imgs, 
                    img_meta=img_metas, 
                    img_refer=img_refers)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)

            if data is None:
                idx = self._rand_another(idx)
                continue
            return data