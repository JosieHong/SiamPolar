'''
@Author: JosieHong
@Date: 2020-05-03 22:33:50
@LastEditTime: 2020-05-10 18:35:10
'''

import os
import glob

import numpy as np
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt

from mmdet.apis import init_detector, inference_tracker, show_result
import mmcv

config_file = '../configs/siampolar/siampolar_r101.py'
checkpoint_file = '../work_dirs/trash/epoch_36.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

imgs = sorted(glob.glob('../data/DAVIS/JPEGImages/480p/car-roundabout/*jpg'))
img_refer = '../data/DAVIS/JPEGImages/480p/car-roundabout/00000.jpg'
# bbox = np.array([96, 214, 369, 221]) # bear
bbox = np.array([160, 78, 382, 313]) # car-roundabout

for i, img in enumerate(imgs):
    result = inference_tracker(model, img, img_refer, bbox)
    outfile = os.path.join('./', str(i)+'.png')
    show_result(img, result, model.CLASSES, score_thr=0.3, show=False, out_file=outfile)
    print("save {}".format(os.path.join('./', str(i)+'.png')))
