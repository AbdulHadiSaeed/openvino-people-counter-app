#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time

import cv2
from openvino.inference_engine import IENetwork, IECore

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side, tiny_flag):
        if( tiny_flag):
            self.num = 3 if 'num' not in param else int(param['num'])
            self.coords = 4 if 'coords' not in param else int(param['coords'])
            self.classes = 80 if 'classes' not in param else int(param['classes'])
            self.side = side
            self.anchors = [10.0, 14.0, 23.0, 27.0, 37.0, 58.0, 81.0, 82.0, 135.0, 169.0, 344.0, 319.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        else:
            self.num = 3 if 'num' not in param else int(param['num'])
            self.coords = 4 if 'coords' not in param else int(param['coords'])
            self.classes = 80 if 'classes' not in param else int(param['classes'])
            self.side = side
            self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                            198.0,
                            373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.



def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union
