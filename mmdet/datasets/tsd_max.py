'''
@Author: JosieHong
@Date: 2020-04-26 12:40:11
@LastEditAuthor: JosieHong
LastEditTime: 2020-08-17 00:00:53
'''

import os.path as osp
import warnings
import math

import mmcv
import numpy as np
from imagecorruptions import corrupt
from mmcv.parallel import DataContainer as DC
import torch

from .utils import random_scale, to_tensor
from .registry import DATASETS
from .coco_seg import Coco_Seg_Dataset, INF

@DATASETS.register_module
class TSD_MAX_Seg_Dataset(Coco_Seg_Dataset):

    CLASSES = ('Section8', 'Section6', 'Section63', 'Section33', 'Section11',
                'Section2', 'Section48', 'Section13', 'Section64', 'Section4',
                'Section75')
                
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 refer_scale=(127,127),
                 num_polar=36,
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
                 test_mode=False,
                 strides=[8, 16, 32, 64, 128],
                 regress_ranges=[(-1, 64), (64, 128), 
                            (128, 256), (256, 512), (512, 1e8)]):
        super(TSD_MAX_Seg_Dataset, self).__init__(ann_file,
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
        self.strides = strides
        self.regress_ranges = regress_ranges
        assert num_polar in [36, 72]
        self.num_polar = num_polar

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix[:-11], img_info['filename']))
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
        img_refer = mmcv.imread(osp.join(self.img_prefix[:-11], refer_info['filename']))
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

        featmap_sizes = self.get_featmap_size(pad_shape) 
        # featmap_sizes: [[32, 32], [16, 16], [8, 8]]

        num_levels = len(self.strides)
        all_level_points = self.get_points(featmap_sizes)
        # level 0 points: torch.Size([1024, 2])
        # level 1 points: torch.Size([256, 2])
        # level 2 points: torch.Size([64, 2])
        
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
            gt_bboxes,gt_masks,gt_labels,concat_points, concat_regress_ranges, self.num_polar)
        
        data['_gt_labels'] = DC(_labels)
        data['_gt_bboxes'] = DC(_bbox_targets)
        data['_gt_masks'] = DC(_mask_targets)
        #--------------------offline ray label generation-----------------------------

        return data

    def get_featmap_size(self, shape):
        h,w = shape[:2]
        featmap_sizes = []
        for i in self.strides:
            featmap_sizes.append([int(h / i)+1, int(w / i)+1])
        return featmap_sizes
        
    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix[:-11], img_info['filename']))
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
        img_refer = mmcv.imread(osp.join(self.img_prefix[:-11], refer_info['filename']))
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
    
    # fit different polar nunbers
    def polar_target_single(self, gt_bboxes, gt_masks, gt_labels, points, regress_ranges, num_polar):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        #xs ys 分别是points的x y坐标
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)   #feature map上所有点对于gtbox的上下左右距离 [num_pix, num_gt, 4]

        #mask targets 也按照这种写 同时labels 得从bbox中心修改成mask 重心
        mask_centers = []
        mask_contours = []
        #第一步 先算重心  return [num_gt, 2]

        for mask in gt_masks:
            cnt, contour = self.get_single_centerpoint(mask)
            contour = contour[0]
            contour = torch.Tensor(contour).float()
            y, x = cnt
            mask_centers.append([x,y])
            mask_contours.append(contour)
        mask_centers = torch.Tensor(mask_centers).float()
        # 把mask_centers assign到不同的层上,根据regress_range和重心的位置
        mask_centers = mask_centers[None].expand(num_points, num_gts, 2)

        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # condition1: inside a gt bbox
        # add center sample
        if self.center_sample:
            if self.use_mask_center:
                inside_gt_bbox_mask = self.get_mask_sample_region(gt_bboxes,
                                                             mask_centers,
                                                             self.strides,
                                                             self.num_points_per_level,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)
            else:
                inside_gt_bbox_mask = self.get_sample_region(gt_bboxes,
                                                             self.strides,
                                                             self.num_points_per_level,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]

        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1])

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0         #[num_gt] 介于0-80

        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        pos_inds = labels.nonzero().reshape(-1)

        mask_targets = torch.zeros(num_points, num_polar).float()
        
        pos_mask_ids = min_area_inds[pos_inds]
        for p,id in zip(pos_inds, pos_mask_ids):
            x, y = points[p]
            pos_mask_contour = mask_contours[id]
            # SiamPolar: interpolate
            new_contour = []
            contour_length = len(pos_mask_contour)
            for i in range(contour_length):
                new_contour.append(pos_mask_contour[i])
                # new_contour.append((3*pos_mask_contour[i]+pos_mask_contour[(i+1)%contour_length])/4)
                new_contour.append((pos_mask_contour[i]+pos_mask_contour[(i+1)%contour_length])/2)
                # new_contour.append((pos_mask_contour[i]+3*pos_mask_contour[(i+1)%contour_length])/4)
            new_pos_mask_contour = torch.cat(new_contour, dim=0).unsqueeze(1)
            # print(pos_mask_contour.size())
            # print(new_pos_mask_contour.size())
            # print(new_pos_mask_contour)
            # exit()

            dists, coords = self.get_coordinates(x, y, new_pos_mask_contour, num_polar)
            mask_targets[p] = dists
        
        return labels, bbox_targets, mask_targets

    def get_coordinates(self, c_x, c_y, pos_mask_contour, num_polar):
        ct = pos_mask_contour[:, 0, :]
        x = ct[:, 0] - c_x
        y = ct[:, 1] - c_y
        # angle = np.arctan2(x, y)*180/np.pi
        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle = angle.int()
        # dist = np.sqrt(x ** 2 + y ** 2)
        dist = torch.sqrt(x ** 2 + y ** 2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]

        # generate num_polar angles
        new_coordinate = {}
        step_size = int(360/num_polar)
        for i in range(0, 360, step_size):
            if i in angle:
                d = dist[angle==i].max()
                new_coordinate[i] = d
            elif i + 1 in angle:
                d = dist[angle == i+1].max()
                new_coordinate[i] = d
            elif i - 1 in angle:
                d = dist[angle == i-1].max()
                new_coordinate[i] = d
            elif i + 2 in angle:
                d = dist[angle == i+2].max()
                new_coordinate[i] = d
            elif i - 2 in angle:
                d = dist[angle == i-2].max()
                new_coordinate[i] = d
            elif i + 3 in angle:
                d = dist[angle == i+3].max()
                new_coordinate[i] = d
            elif i - 3 in angle:
                d = dist[angle == i-3].max()
                new_coordinate[i] = d
            # josie.add
            elif i + 4 in angle:
                d = dist[angle == i+4].max()
                new_coordinate[i] = d
            elif i - 4 in angle:
                d = dist[angle == i-4].max()
                new_coordinate[i] = d
            elif i + 5 in angle:
                d = dist[angle == i+5].max()
                new_coordinate[i] = d
            elif i - 5 in angle:
                d = dist[angle == i-5].max()
                new_coordinate[i] = d

        distances = torch.zeros(num_polar)

        for a in range(0, 360, step_size):
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[a//step_size] = 1e-6
            else:
                distances[a//step_size] = new_coordinate[a]

        return distances, new_coordinate

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)

            if data is None:
                idx = self._rand_another(idx)
                continue
            return data