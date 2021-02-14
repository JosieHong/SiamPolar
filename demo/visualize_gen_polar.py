'''
Author: JosieHong
Date: 2020-10-03 00:30:20
LastEditAuthor: JosieHong
LastEditTime: 2020-10-03 03:00:18
Note: 
    python visualize_gen_polar.py ../imgs/polar_vis/ 00013 72
    python visualize_gen_polar.py ../imgs/polar_vis/ 00167c 72
    python visualize_gen_polar.py ../imgs/polar_vis/ 00512c 72
    python visualize_gen_polar.py ../imgs/polar_vis/ 00867c 72
    python visualize_gen_polar.py ../imgs/polar_vis/ 01583c 72
'''
import cv2
import torch
import numpy as np
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

def get_single_centerpoint(mask):
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) # only save the biggest one
    '''debug IndexError: list index out of range'''
    count = contour[0][:, 0, :]
    try:
        center = get_centerpoint(count)
    except:
        x,y = count.mean(axis=0)
        center=[int(x), int(y)]

    # max_points = 360
    # if len(contour[0]) > max_points:
    #     compress_rate = len(contour[0]) // max_points
    #     contour[0] = contour[0][::compress_rate, ...]
    return center, contour

def get_coordinates(c_x, c_y, pos_mask_contour, polar_num):
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

path_to_imgs = sys.argv[1]
name = sys.argv[2] 
polar_num = int(sys.argv[3])
mask_path = path_to_imgs + name+".png"
img_path = path_to_imgs + name+".jpg"
result_path = path_to_imgs + name+"_"+str(polar_num)+"vis.png"

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path)

center, contour = get_single_centerpoint(mask)
contour = torch.Tensor(contour[0]).float()
center = tuple(center)

distances, coordination = get_coordinates(center[0], center[1], contour, polar_num)
print(distances, coordination)

pts = []
for k in coordination.keys():
    ptEnd = (int(center[0]+int(coordination[k])*np.sin(k*np.pi/180)), int(center[1]+int(coordination[k])*np.cos(k*np.pi/180)))
    pts.append(ptEnd)
    point_color = (0, 140, 255) # BGR 
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

cv2.polylines(img, [pts], True, (224, 255, 255), 1)
# cv2.circle(img, center, 3, (224, 255, 255), 4)
cv2.putText(img,'IoU: {:0.3f}'.format(iou), (50,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 140, 255), 4)
cv2.imwrite(result_path, img)
print("save {}".format(result_path))