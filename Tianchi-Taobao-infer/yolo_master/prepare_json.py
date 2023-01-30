import os
from os.path import sep, join, splitext
import json
from tqdm import tqdm

import cv2
import imagesize


def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')

def get_img_dicts(input_dir):
    image_dict = {}
    image_dict['file_name'] =input_dir # pjoin(input_dir, item_id, img)
    w, h = imagesize.get(image_dict['file_name'])
    image_dict['height'] = h
    image_dict['width'] = w
    image_dict['item_id'] = input_dir.split('/')[-2]
    image_dict['type'] = 'image'

    return image_dict

def get_vid_dicts(input_dir):

    image_dict = {}
    image_dict['file_name'] =input_dir # pjoin(input_dir, item_id, img)
    capture = cv2.VideoCapture(image_dict['file_name'])
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_dict['height'] = h
    image_dict['width'] = w
    image_dict['item_id'] = input_dir.split('/')[-1].split('.')[0]
    image_dict['type'] = 'video'

    return image_dict

# d = get_img_dicts('../input/validation_dataset_part1')