'''
@Author: JosieHong
@Date: 2020-05-05 00:47:49
LastEditTime: 2021-06-24 23:45:08
'''

# model settings
model = dict(
    type='SiamPolar',
    pretrained='open-mmlab://resnet101_caffe',
    backbone=dict(
        type='SiamResNet',
        depth=101, 
        template_depth=50,
        template_pretrained='open-mmlab://resnet50_caffe',
        num_stages=4,
        strides=(1, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe',
        correlation_blocks=[5], # block index
        # attention_blocks=[2, 3, 4]
        ), 
    neck=dict(
        type='Single_Connect',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        out_blocks=[5]),
    bbox_head=dict(
        type='SiamPolar_Head', 
        num_classes=120,
        num_polar=36,
        in_channels=256,
        stacked_convs=1, 
        feat_channels=256,
        strides=[32], 
        regress_ranges=[(-1, 1e8)],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_mask=dict(type='MaskIOULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'DAVIS_Seg_Dataset'
data_root = 'data/DAVIS/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
data = dict(
    imgs_per_gpu=16, 
    workers_per_gpu=8, 
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/480p_trainval.json',
        img_prefix=data_root,
        img_scale=(255, 255), # original size
        img_norm_cfg=img_norm_cfg,
        refer_scale=(127, 127),
        num_polar=36,
        # size_divisor=0,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False,
        # for Single_Connect
        strides=[32], 
        regress_ranges=[(-1, 1e8)]),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/480p_val.json',
        img_prefix=data_root,
        img_scale=(255, 255),
        img_norm_cfg=img_norm_cfg,
        refer_scale=(127, 127),
        num_polar=36,
        # size_divisor=0,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/480p_val.json',
        img_prefix=data_root,
        img_scale=(255, 255),
        img_norm_cfg=img_norm_cfg,
        refer_scale=(127, 127),
        num_polar=36,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        resize_keep_ratio=False,
        test_mode=True))
# optimizer
lr_ratio = 1

optimizer = dict(
    type='SGD',
    lr=0.01 * lr_ratio,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3 / lr_ratio,
    step=[8, 11, 17, 23])
checkpoint_config = dict(interval=1)
# for training on colab, which doesn't support os.symlink()
# checkpoint_config = dict(interval=1, create_symlink=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/trash'
load_from = None
resume_from = None
workflow = [('train', 1)]
