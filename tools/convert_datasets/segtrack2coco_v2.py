'''
Author: JosieHong
Date: 2021-01-12 22:13:28
LastEditAuthor: JosieHong
LastEditTime: 2021-01-17 01:02:08
'''
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import json, yaml
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask


class SegTrack():
    """
        SegTrack class to convert annotations to COCO Json format
    """
    def __init__(self, datapath):
        self.info = {"year" : 2011,
                     "version" : "1.0",
                     "description" : "A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation (SegTrack)",
                     "contributor" : "David Tsai and Matthew Flagg and James M.Rehg",
                     "url" : "http://cpl.cc.gatech.edu/projects/SegTrack/",
                     "date_created" : "2021"
                    }
        self.licenses = [{"id": 1,
                          "name": "Attribution-NonCommercial",
                          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                         }]
        self.type = "instances"
        self.datapath = datapath
        catpaths = os.listdir(os.path.join(datapath, "JPEGImages"))
        print(catpaths)
        self.categories = [{"id": seq_id+1, "name": seq_name, "supercategory": seq_name}
                            for seq_id, seq_name in enumerate(catpaths)]
        self.cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(self.categories)}

        image_set = []
        for catpath in catpaths:
            imlist = []
            for impath in os.listdir(os.path.join(datapath, "JPEGImages", catpath)):
                imlist.append(os.path.join("JPEGImages", catpath, impath))
            annotlist = []
            for annotpath in os.listdir(os.path.join(datapath, "GroundTruth", catpath)):
                annotlist.append(os.path.join("GroundTruth", catpath, annotpath))
            imlist.sort()
            annotlist.sort()
            image_set += zip(imlist, annotlist)

        images, annotations = self.__get_image_annotation_pairs__(image_set)
        json_data = {"info" : self.info,
                        "images" : images,
                        "licenses" : self.licenses,
                        "type" : self.type,
                        "annotations" : annotations,
                        "categories" : self.categories}

        with open(os.path.join(self.datapath, "val.json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
        print("Save the annotation in {}".format(os.path.join(self.datapath, "val.json")))

    def __get_image_annotation_pairs__(self, image_set):
        images = []
        annotations = []
        flag_name = None
        first_frame = None
        for imId, paths in enumerate(image_set):
            impath, annotpath = paths[0], paths[1]
            print(impath, annotpath)
            name = impath.split("/")[1]
            # get the first frame's id
            if name != flag_name:
                first_frame = imId+1
                flag_name = name
            
            img = np.array(Image.open(os.path.join(self.datapath, impath)).convert('RGB'))
            mask = np.array(Image.open(os.path.join(self.datapath, annotpath)).convert('L'))
            if np.all(mask == 0):
                continue

            segmentation, bbox, area = self.__get_annotation__(mask, img)
            images.append({"date_captured" : "2016",
                           "file_name" : impath, 
                           "id" : imId+1,
                           "license" : 1,
                           "url" : "",
                           "height" : mask.shape[0],
                           "width" : mask.shape[1],
                           "first_frame": first_frame})
            annotations.append({"segmentation" : segmentation,
                                "area" : np.float(area),
                                "iscrowd" : 0,
                                "image_id" : imId+1,
                                "bbox" : bbox,
                                "category_id" : self.cat2id[name],
                                "id": imId+1})
        return images, annotations

    def __get_annotation__(self, mask, image=None):

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        RLE = cocomask.merge(RLEs)
        # RLE = cocomask.encode(np.asfortranarray(mask))
        area = cocomask.area(RLE)
        [x, y, w, h] = cv2.boundingRect(mask)

        # show
        # if image is not None:
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     cv2.drawContours(image, contours, -1, (0,255,0), 1)
        #     cv2.rectangle(image,(x,y),(x+w,y+h), (255,0,0), 2)
        #     cv2.imshow("", image)
        #     cv2.waitKey(1)

        return segmentation, [x, y, w, h], area

if __name__ == "__main__":
    datapath = "/data/dataset/SegTrackv2/"
    # datapath = sys.argv[1]
    SegTrack(datapath)