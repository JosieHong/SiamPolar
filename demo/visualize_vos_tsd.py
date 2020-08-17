'''
@Author: JosieHong
@Date: 2020-05-03 22:33:50
LastEditTime: 2020-08-17 00:51:57
'''

import os
import glob

import numpy as np
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt

from mmdet.apis import init_detector, inference_tracker, show_result
import mmcv

config_file = '../configs/siampolar/siampolar_r101_tsd-max.py'
checkpoint_file = '../work_dirs/trash/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

inputs = {  
            'Section11': [324, 333, 190+324, 128+333],
            'Section13': [245, 318, 250+245, 193+318],
            'Section2': [862, 370, 167+862, 166+370],
            'Section33': [241, 179, 271+241, 293+179], 
            'Section4': [476, 206, 118+476, 170+206],
            'Section48': [369, 330, 179+369, 126+330],
            'Section6': [896, 334, 280+896, 208+334],
            'Section63': [172, 319, 283+172, 195+319],
            'Section64': [29, 279, 509+29, 334+279],
            'Section75': [857, 215, 310+857, 188+215],
            'Section8': [836, 521, 391+836, 295+521]
        }
first_frame = {
            'Section11': '00330c.jpg',
            'Section13': '00166c.jpg',
            'Section2': '01583c.jpg',
            'Section33': '00735c.jpg', 
            'Section4': '00512c.jpg',
            'Section48': '00927c.jpg',
            'Section6': '00000c.jpg',
            'Section63': '00601c.jpg',
            'Section64': '00992c.jpg',
            'Section75': '00065c.jpg',
            'Section8': '01036c.jpg'
}

for video_name in inputs.keys():
    imgs = sorted(glob.glob(os.path.join('../data/TSD-MAX_VOS/JPEGImages/1080p', video_name, '*jpg')))
    img_refer = os.path.join('../data/TSD-MAX_VOS/JPEGImages/1080p/', video_name, first_frame[video_name])
    for i, img in enumerate(imgs):
        result = inference_tracker(model, img, img_refer, np.array(inputs[video_name]))
        outfile = os.path.join('/data/hyh/SiamPolar/demo/vis/', video_name, str(i)+'.png')
        show_result(img, result, model.CLASSES, score_thr=0.3, show=False, out_file=outfile)
        print("save {}".format(outfile))
