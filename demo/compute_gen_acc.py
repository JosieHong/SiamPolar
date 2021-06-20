'''
Date: 2021-05-26 14:36:22
LastEditors: yuhhong
LastEditTime: 2021-05-30 15:31:14
Note: 
    python compute_gen_acc.py 0 > gen_acc0_sim.out
    python compute_gen_acc.py 1 > gen_acc1_sim.out
    python compute_gen_acc.py 2 > gen_acc2_sim.out
    python compute_gen_acc.py 3 > gen_acc3_sim.out
    python compute_gen_acc.py 4 > gen_acc4_sim.out
'''
import sys
import os
import cv2
import numpy as np
import torch

from visualize_gen_polar import get_single_centerpoint, interpolat_contour, get_coordinates

def compute_acc(img_path, mask_path, inter_num=1, polar_num=36):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    if mask.any() == None or img.any() == None:
        print("The image({}) and the mask({}) is none.".format(img_path, mask_path))
        exit()

    center, contour = get_single_centerpoint(mask)
    if contour == None:
        print("The contour of image({}) and the mask({}) is none.".format(img_path, mask_path))
        exit()
        
    center = tuple(center)
    contour = torch.Tensor(contour[0]).float()
    
    # interpolation
    for i in range(inter_num): 
        contour = interpolat_contour(contour)

    _, coordination = get_coordinates(center[0], center[1], contour, polar_num)

    pts = []
    for k in coordination.keys():
        ptEnd = (int(center[0]+int(coordination[k])*np.sin(k*np.pi/180)), int(center[1]+int(coordination[k])*np.cos(k*np.pi/180)))
        pts.append(ptEnd)
        
        # # point_color = (0, 140, 255) # BGR 
        # point_color = (237, 149, 100) # BGR 
        # thickness = 1
        # lineType = 4
        # # print(center)
        # # print(ptEnd)
        # cv2.line(img, center, ptEnd, point_color, thickness, lineType)

    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))

    # IoU calculation
    mask_ori = np.where(mask>1, 1, 0)
    mask_gen = np.zeros(img.shape, dtype=np.float32)
    mask_gen = cv2.drawContours(mask_gen, [pts], -1, (0,255,0), -1)
    mask_gen = cv2.cvtColor(mask_gen, cv2.COLOR_BGR2GRAY)
    mask_gen = np.where(mask_gen>1, 1, 0)
    iou = np.sum(np.logical_and(mask_ori, mask_gen)) / np.sum(np.logical_or(mask_ori, mask_gen))
    print("{}: {}".format(mask_path, iou))
    return iou



if __name__ == '__main__':
    data_path = "../data/DAVIS/"
    video_names = os.listdir(os.path.join(data_path, "Annotations", "1080p"))
    iou = 0
    frame = 0
    for v in video_names:
        if v in ['bmx-bumps', 'surf']: 
            continue
        # if v not in ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']:
        #     continue
        # if v not in ['rhino']:
        #     continue
        # if v not in ['bmx-trees', 'bus', 'horsejump-high', 'scooter-gray', 'rhino', 'soccerball', 'swing']:
        #     continue

        mask_names = os.listdir(os.path.join(data_path, "Annotations", "1080p", v))
        img_names = os.listdir(os.path.join(data_path, "JPEGImages", "1080p", v))
        # print("mask_names: {}, img_names: {}".format(mask_names, img_names))
        for m, i in zip(mask_names, img_names):
            mask_path = os.path.join(data_path, "Annotations", "1080p", v, m)
            img_path = os.path.join(data_path, "JPEGImages", "1080p", v, i)
            # print("mask_path: {}, img_path: {}".format(mask_path, img_path))
            
            iou += compute_acc(img_path, mask_path, inter_num=int(sys.argv[1]))
            frame += 1
    print("avg IoU: {}".format(iou/frame))