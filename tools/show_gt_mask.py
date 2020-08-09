'''
@Author: JosieHong
@Date: 2020-06-29 10:58:12
@LastEditAuthor: JosieHong
@LastEditTime: 2020-06-30 01:42:22
'''
import os
import numpy as np

import torch
import cv2

# davis.py
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

def get_centerpoint(lis):
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

def get_coordinates(c_x, c_y, pos_mask_contour, num_polar):
    
    ct = pos_mask_contour[:, 0, :]
    print(ct)
    x = ct[:, 0] - c_x
    y = ct[:, 1] - c_y
    print(x)
    print(y)
    print(len(x))
    print(len(y))
    
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

    distances = torch.zeros(num_polar)

    for a in range(0, 360, step_size):
        if not a in new_coordinate.keys():
            new_coordinate[a] = torch.tensor(1e-6)
            distances[a//step_size] = 1e-6
        else:
            distances[a//step_size] = new_coordinate[a]

    return distances, new_coordinate

def distance2coord(points, distances, angles, num_polar):
    '''Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
        angles (Tensor):
        max_shape (tuple): Shape of the image.

    Returns:
        Coordinates(list): [x, y].
    '''
    c_x, c_y = points[0], points[1]
    c_x = c_x.repeat(num_polar)
    c_y = c_y.repeat(num_polar)
    
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    x = distances * sin + c_x
    y = distances * cos + c_y
   
    points = [[i_x, i_y] for i_x, i_y in zip(x, y)]

    return points

def main():
    num_polar = 36
    data_path = './data/DAVIS'
    img_name = 'car-shadow/00000.jpg'
    mask_name = 'car-shadow/00000.png'

    img = cv2.imread(os.path.join(data_path, 'JPEGImages/480p', img_name))
    mask = cv2.imread(os.path.join(data_path, 'Annotations/480p', mask_name))  
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cnt, contour = get_single_centerpoint(mask)
    cv2.drawContours(img, contour, -1, (255, 0, 0), 3)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

    contour = contour[0]
    contour = torch.Tensor(contour).float()
    c_y, c_x = cnt
    dists, coords = get_coordinates(c_x, c_y, contour, num_polar)
    print(coords)
    points = distance2coord(torch.Tensor([c_x, c_y]), dists, torch.Tensor(list(coords.keys())), num_polar)
    # print(points)
    # print(img.shape)

    print('center: ', tuple(cnt))
    for i, point in enumerate(points): 
        print('point {}: {}'.format(i, point))

    for point in points: 
        cv2.line(img, tuple(cnt), tuple(point), color=(0, 0, 255))
    cv2.imshow('image', img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    

    
    
    
    