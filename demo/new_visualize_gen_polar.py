'''
Author: JosieHong
Date: 2020-10-03 00:30:20
LastEditAuthor: JosieHong
LastEditTime: 2021-05-30 15:34:50
Note: 
    python new_visualize_gen_polar.py ../imgs/polar_vis/ 00065 36 0
    python new_visualize_gen_polar.py ../imgs/polar_vis/ 00006 36 0
    python visualize_gen_polar.py ../imgs/polar_vis/ 00065 36 0
'''
import cv2
import torch
import numpy as np
import math
import sys

def get_centerpoint(self, lis):
    area = 0.0
    x, y = 0.0, 0.0
    a = len(lis)
    for i in range(a):
        lat = lis[i][0]
        lng = lis[i][1]
        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]
        else:
            lat1 = lis[i - 1][0]
            lng1 = lis[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0
    x = x / area
    y = y / area

    return [int(x), int(y)]

def merge_contours(contours): 
    alpha = 0.25
    
    # init
    b = contours[0][:, 0, :]
    cx, cy = b.mean(axis=0)
    # guarantee that the threshold is at the same level as the object size
    # thrx = contours[0][:, 0, :][:, 0].max() - contours[0][:, 0, :][:, 0].min()
    # thry = contours[0][:, 0, :][:, 1].max() - contours[0][:, 0, :][:, 1].min()
    records = [0 for i in range(len(contours))]
    new_contours = [contours[0]]
    records[0] = 1

    flag = True
    while (flag == True):
        flag = False
        for i in range(1, len(contours)-1): 
            tmp = contours[i][:, 0, :]
            tx, ty = tmp.mean(axis=0)
            if records[i] == 0:
                d = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                lx = b[:, 0].max() - b[:, 0].min() + tmp[:, 0].max() - tmp[:, 0].min()
                ly = b[:, 1].max() - b[:, 1].min() + tmp[:, 1].max() - tmp[:, 1].min()
                l = math.sqrt(lx ** 2 + ly ** 2)
                # print("d: {}, l: {}".format(d, l))
                if d <= alpha * l:
                    # print("Add a new contour!")
                    new_contours.append(contours[i])
                    records[i] = 1
                    flag = True
                    cx = (cx + tx) / 2
                    cy = (cy + ty) / 2
                
    return new_contours

def get_single_centerpoint(mask):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True) # only save the biggest one
    '''debug IndexError: list index out of range'''
    if len(contours) == 0:
        return None, None

    count = contours[0][:, 0, :]
    try:
        center = get_centerpoint(count)
    except:
        x,y = count.mean(axis=0)
        center = [int(x), int(y)]
        
    if len(contours) > 1: 
        # keep the contours near the biggest contour
        new_contours = merge_contours(contours)
    else:
        new_contours = [contours[0]] # the biggest contour

    return center, new_contours

def get_coordinates(c_x, c_y, pos_mask_contours, polar_num):
    for i in range(len(pos_mask_contours)):
        if i == 0:
            ct_x = pos_mask_contours[i][:, 0, :][:, 0]
            ct_y = pos_mask_contours[i][:, 0, :][:, 1]
        else:
            ct_x = np.append(ct_x, pos_mask_contours[i][:, 0, :][:, 0])
            ct_y = np.append(ct_y, pos_mask_contours[i][:, 0, :][:, 1])
    ct_x = torch.Tensor(ct_x).float()
    ct_y = torch.Tensor(ct_y).float()
    x = ct_x - c_x
    y = ct_y - c_y
    # angle = np.arctan2(x, y)*180/np.pi
    angle = torch.atan2(x, y) * 180 / np.pi
    angle[angle < 0] += 360
    angle = angle.int()
    # dist = np.sqrt(x ** 2 + y ** 2)
    dist = torch.sqrt(x ** 2 + y ** 2)
    angle, idx = torch.sort(angle)
    dist = dist[idx]

    # generate 36 angles
    new_coordinate = {}
    step = 360 // polar_num
    for i in range(0, 360, step):
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

    distances = torch.zeros(polar_num)

    for a in range(0, 360, step):
        if not a in new_coordinate.keys():
            new_coordinate[a] = torch.tensor(1e-6)
            distances[a//step] = 1e-6
        else:
            distances[a//step] = new_coordinate[a]

    return distances, new_coordinate



if __name__ == '__main__':
    path_to_imgs = sys.argv[1]
    name = sys.argv[2] 
    polar_num = int(sys.argv[3])
    inter_flag = int(sys.argv[4])
    mask_path = path_to_imgs + name+".png"
    img_path = path_to_imgs + name+".jpg"
    if inter_flag:
        result_path = path_to_imgs + name+"_"+str(polar_num)+"vis_wi.png"
    else:
        result_path = path_to_imgs + name+"_"+str(polar_num)+"vis_wo.png"

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)

    center, contours = get_single_centerpoint(mask)
    # contours = torch.Tensor(contours).float()
    center = tuple(center)

    # interpolation
    # if inter_flag:
    #     contour = interpolat_contour(contour)

    distances, coordination = get_coordinates(center[0], center[1], contours, polar_num)
    print(distances, coordination)

    pts = []
    for k in coordination.keys():
        ptEnd = (int(center[0]+int(coordination[k])*np.sin(k*np.pi/180)), int(center[1]+int(coordination[k])*np.cos(k*np.pi/180)))
        pts.append(ptEnd)
        # point_color = (0, 140, 255) # BGR 
        point_color = (237, 149, 100) # BGR 
        thickness = 1
        lineType = 4
        # print(center)
        # print(ptEnd)
        cv2.line(img, center, ptEnd, point_color, thickness, lineType)

    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))

    # IoU calculation
    mask_ori = np.where(mask>1, 1, 0)
    mask_gen = np.zeros(img.shape, dtype=np.float32)
    mask_gen = cv2.drawContours(mask_gen, [pts], -1, (0,255,0), -1)
    mask_gen = cv2.cvtColor(mask_gen, cv2.COLOR_BGR2GRAY)
    mask_gen = np.where(mask_gen>1, 1, 0)
    iou = np.sum(np.logical_and(mask_ori, mask_gen)) / np.sum(np.logical_or(mask_ori, mask_gen))
    print("IoU = {}".format(iou))

    cv2.polylines(img, [pts], True, (224, 255, 255), 2)
    # center
    # cv2.circle(img, center, 3, (224, 255, 255), 4)
    # IoU
    # cv2.putText(img,'IoU: {:0.3f}'.format(iou), (50,150), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 140, 255), 6) # Davis
    cv2.putText(img,'IoU: {:0.3f}'.format(iou), (50,150), cv2.FONT_HERSHEY_COMPLEX, 2, (225, 105, 65), 4) # TSD-max
    # cv2.putText(img,'IoU: {:0.3f}'.format(iou), (30,300), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 140, 255), 2) # small
    # cv2.putText(img,'IoU: {:0.3f}'.format(iou), (50,1000), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 140, 255), 6) # bottom
    cv2.imwrite(result_path, img)
    print("save {}".format(result_path))