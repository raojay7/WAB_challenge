import os
from os.path import sep, join, splitext
import json
import copy
from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2
import imagesize

from logger import create_folder

DEBUG = False
aspect_template = np.array([0.25, 0.335, 0.415, 0.5, 0.5721925, 0.66857143, 0.8, 1., 1.25, 1.4957264, 1.74766355, 2.])


def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')


label2cat = {
    "short sleeve top": 0,
    "long sleeve top": 1,
    "short sleeve shirt": 2,
    "long sleeve shirt": 3,
    "vest top": 4,
    "sling top": 5,
    "sleeveless top": 6,
    "short outwear": 7,
    "short vest": 8,
    "long sleeve dress": 9,
    "short sleeve dress": 10,
    "sleeveless dress": 11,
    "long vest": 12,
    "long outwear": 13,
    "bodysuit": 14,
    "classical": 15,
    "short skirt": 16,
    "medium skirt": 17,
    "long skirt": 18,
    "shorts": 19,
    "medium shorts": 20,
    "trousers": 21,
    "overalls": 22
    # '短袖上衣': 0,
    # '长袖上衣': 1,
    # '短袖衬衫': 2,
    # '长袖衬衫': 3,
    # '背心上衣': 4,
    # '吊带上衣': 5,
    # '无袖上衣': 6,
    # '短外套': 7,
    # '短马甲': 8,
    # '长袖连衣裙':  9,
    # '短袖连衣裙': 10,
    # '无袖连衣裙': 11,
    # '长马甲': 12,
    # '长外套': 13,
    # '连体衣': 14,
    # '古风': 15,
    # '古装': 15,
    # '短裙': 16,
    # '中等半身裙': 17,
    # '长半身裙': 18,
    # '短裤': 19,
    # '中裤': 20,
    # '长裤': 21,
    # '背带裤': 22
}


def get_item_dicts_img(input_dir, item_id):
    image_dir = pjoin(input_dir, 'image', item_id)

    json_dir = pjoin(input_dir, 'image_annotation', item_id)
    json_list = os.listdir(json_dir)

    result = []
    for j in json_list:
        with open(pjoin(json_dir, j)) as f:
            d = json.load(f)

        image_dict = {}
        image_dict['file_name'] = pjoin(image_dir, d['img_name'])
        w, h = imagesize.get(image_dict['file_name'])
        image_dict['height'] = h
        image_dict['width'] = w
        image_dict['image_id'] = d['item_id'] + '_' + d['img_name']
        image_dict['type'] = 'image'
        image_dict['item_id'] = d['item_id']

        annotations = []
        for instance in d['annotations']:
            annotations.append({
                'display': instance['display'],
                'bbox': instance['box'],
                'category_id': label2cat[instance['label']],
                'viewpoint': instance['viewpoint'],
                'instance_id': instance['instance_id'],
                'bbox_aspect': (instance['box'][2] - instance['box'][0]) / (instance['box'][3] - instance['box'][1])
            })
        image_dict['annotations'] = annotations
        result.append(image_dict)
    return result


def get_item_dicts_vid(input_dir, item_id):
    image_dir = pjoin(input_dir, 'video', item_id + '.mp4')
    json_dir = pjoin(input_dir, 'video_annotation', item_id + '.json')

    with open(json_dir) as f:
        d = json.load(f)

    capture = cv2.VideoCapture(image_dir)
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result = []
    for j in d['frames']:

        image_dict = {}
        image_dict['file_name'] = image_dir
        image_dict['height'] = h
        image_dict['width'] = w
        image_dict['image_id'] = d['video_id'] + '_' + str(j['frame_index'])
        image_dict['frame'] = j['frame_index']
        image_dict['type'] = 'video'
        image_dict['item_id'] = d['video_id']

        annotations = []
        for instance in j['annotations']:
            annotations.append({
                'display': instance['display'],
                'bbox': instance['box'],
                'category_id': label2cat[instance['label']],
                'viewpoint': instance['viewpoint'],
                'instance_id': instance['instance_id'],
                'bbox_aspect': (instance['box'][2] - instance['box'][0]) / (instance['box'][3] - instance['box'][1])
            })
        image_dict['annotations'] = annotations
        result.append(image_dict)
    return result


def get_dicts(input_dir, item_id_list, type, n_items=None):
    if type == 'image':
        func = get_item_dicts_img
    elif type == 'video':
        func = get_item_dicts_vid

    # 改成这个itemlist为直接输入
    if n_items is not None:
        item_id_list = item_id_list[:n_items]

    result = []
    for item_id in tqdm(item_id_list):
        # 找到对应id的东西
        result += func(input_dir, item_id)

    return result


def split_train_dev(data_dir:str,dev_size:float=0.2):
    """
    随机划分训练集和验证集
    :param data_dir: 训练数据的图片文件目录地址
    :param dev_size: dev集比例
    :return: list_1 = [train_id1, train_id_2, ...],
             list_2 = [dev_id_1, dev_id_2, ...]
    """
    item_id_list = os.listdir(pjoin(data_dir, 'image_annotation'))#itemid的list
    dev_size = int(dev_size * len(item_id_list))
    train_split = item_id_list[dev_size:]
    dev_split = item_id_list[:dev_size]
    return train_split, dev_split
def getMultiSplit(num_dataset=2):
    """
    返回多个整合的分割数据集
    Returns:
    """
    train_data_splits=[]
    dev_data_splits=[]
    for i in range(num_dataset):
        train_split, dev_split=split_train_dev("/root/disk7/leo.gb/TianchiDataset/train_dataset_7w/train_dataset_part"+str(i+1))
        train_data_splits.append(train_split)
        dev_data_splits.append(dev_split)
    return train_data_splits,dev_data_splits

def get_dicts_train_all_img(train_splits):
    pre = "/root/disk7/leo.gb/TianchiDataset/train_dataset_7w/"
    img_result = []
    for i in range(len(train_splits)):
        img_result.extend(get_dicts(pre + 'train_dataset_part' + str(i+1), train_splits[i], 'image'))
    return img_result
def get_dicts_train_all_vid(train_splits):
    pre = "/root/disk7/leo.gb/TianchiDataset/train_dataset_7w/"
    vid_result = []
    for i in range(len(train_splits)):
        vid_result.extend(get_dicts(pre + 'train_dataset_part' + str(i+1), train_splits[i], 'video'))
    return vid_result

def get_dicts_val_all_vid(dev_splits):
    pre = "/root/disk7/leo.gb/TianchiDataset/train_dataset_7w/"
    vid_result = []
    for i in range(len(dev_splits)):
        vid_result.extend(get_dicts(pre + 'train_dataset_part' + str(i + 1), dev_splits[i], 'video'))
    return vid_result

def get_dicts_val_all_img(dev_splits):
    pre = "/root/disk7/leo.gb/TianchiDataset/train_dataset_7w/"
    img_result = []
    for i in range(len(dev_splits)):
        img_result.extend(get_dicts(pre + 'train_dataset_part' + str(i + 1), dev_splits[i], 'image'))
    return img_result


# instance json helpers
def get_instance_dict_img(metric_json):
    result = []
    for d in tqdm(metric_json):
        img_dict = {k: v for k, v in d.items() if k in ('file_name', 'height', 'width', 'image_id', 'type', 'item_id')}
        for ins in d['annotations']:
            if ins['instance_id'] == 0:
                continue
            ins_dict = copy.deepcopy(img_dict)
            ins_dict.update(ins)
            ins_dict['aspect_group'] = int(np.abs(ins_dict['bbox_aspect'] - aspect_template).argmin())
            result.append(ins_dict)
    return result


def get_instance_dict_vid(metric_json):
    result = []
    for d in tqdm(metric_json):
        if d['frame'] < 80:
            continue
        img_dict = {k: v for k, v in d.items() if
                    k in ('file_name', 'height', 'width', 'image_id', 'frame', 'type', 'item_id')}
        for ins in d['annotations']:
            if ins['instance_id'] == 0:
                continue
            ins_dict = copy.deepcopy(img_dict)
            ins_dict.update(ins)
            ins_dict['aspect_group'] = int(np.abs(ins_dict['bbox_aspect'] - aspect_template).argmin())
            result.append(ins_dict)
    return result


def main():
    train_splits,dev_splits=getMultiSplit(num_dataset=5)
    # np.save('train_data_splits.npy', train_data_splits)
    # np.save('dev_data_splits.npy', dev_data_splits)
    # dev_splits = np.load('dev_data_splits.npy')
    # train_splits = np.load('train_data_splits.npy')
    taobao_train_img = get_dicts_train_all_img(train_splits)
    taobao_train_vid = get_dicts_train_all_vid(train_splits)

    taobao_val_img = get_dicts_val_all_img(dev_splits)
    taobao_val_vid = get_dicts_val_all_vid(dev_splits)

    create_folder('../myspace/input')

    with open('../myspace/input/taobao_train_img_metric.json', 'w') as f:
        json.dump(taobao_train_img, f)

    with open('../myspace/input/taobao_val_img_metric.json', 'w') as f:
        json.dump(taobao_val_img, f)

    with open('../myspace/input/taobao_train_vid_metric.json', 'w') as f:
        json.dump(taobao_train_vid, f)

    with open('../myspace/input/taobao_val_vid_metric.json', 'w') as f:
        json.dump(taobao_val_vid, f)

    # with open('/myspace/input/taobao_train_img_metric.json', 'r') as f:
    #     taobao_train_img = json.load(f)

    # with open('/myspace/input/taobao_val_img_metric.json', 'r') as f:
    #     taobao_val_img = json.load(f)

    # with open('/myspace/input/taobao_train_vid_metric.json', 'r') as f:
    #     taobao_train_vid = json.load(f)

    # with open('/myspace/input/taobao_val_vid_metric.json', 'r') as f:
    #     taobao_val_vid = json.load(f)

    train_img_inst = get_instance_dict_img(taobao_train_img)
    train_vid_inst = get_instance_dict_vid(taobao_train_vid)

    train_img_inst_df = pd.DataFrame.from_dict(train_img_inst)
    train_vid_inst_df = pd.DataFrame.from_dict(train_vid_inst)

    img_inst_ids = train_img_inst_df['instance_id'].unique()
    print('No. instances in round 1 images: {}'.format(len(img_inst_ids)))
    vid_inst_ids = train_vid_inst_df['instance_id'].unique()
    print('No. instances in round 1 videos: {}'.format(len(vid_inst_ids)))
    inst_ids = np.union1d(img_inst_ids, vid_inst_ids)
    print('Total no. unique instances in round 1: {}'.format(len(inst_ids)))

    # encode instance ids
    instance_encoder = {inst_id: i for i, inst_id in enumerate(inst_ids)}
    train_img_inst_df['instance_id'] = train_img_inst_df['instance_id'].map(instance_encoder)
    train_vid_inst_df['instance_id'] = train_vid_inst_df['instance_id'].map(instance_encoder)

    train_img_inst = train_img_inst_df.to_dict(orient='records')
    train_vid_inst = train_vid_inst_df.to_dict(orient='records')

    with open('../myspace/input/taobao_round1_img_inst_80.json', 'w') as f:
        json.dump(train_img_inst, f)

    with open('../myspace/input/taobao_round1_vid_inst_80.json', 'w') as f:
        json.dump(train_vid_inst, f)

    # save encoder map
    instance_encoder = pd.DataFrame.from_dict(instance_encoder, orient='index').reset_index()
    instance_encoder.columns = ['instance_id', 'instance_cat']
    instance_encoder.to_feather('../myspace/input/taobao_round1_instance_encoder_80.feather')

    # round 2 data
    val_img_inst = get_instance_dict_img(taobao_val_img)
    val_vid_inst = get_instance_dict_vid(taobao_val_vid)

    with open('../myspace/input/taobao_round2_img_inst_80.json', 'w') as f:
        json.dump(val_img_inst, f)

    with open('../myspace/input/taobao_round2_vid_inst_80.json', 'w') as f:
        json.dump(val_vid_inst, f)

    val_img_inst_df = pd.DataFrame.from_dict(val_img_inst)
    val_vid_inst_df = pd.DataFrame.from_dict(val_vid_inst)

    print('No. instances in round 2 images: {}'.format(val_img_inst_df['instance_id'].nunique()))
    print('No. instances in round 2 videos: {}'.format(val_vid_inst_df['instance_id'].nunique()))

    shared_inst = np.intersect1d(val_img_inst_df['instance_id'], val_vid_inst_df['instance_id'])
    print('No. shared instances in round 2: {}'.format(len(shared_inst)))

    img_sub_df = val_img_inst_df[val_img_inst_df['instance_id'].isin(shared_inst[:3000])]
    vid_sub_df = val_vid_inst_df[val_vid_inst_df['instance_id'].isin(shared_inst[:3000])]

    with open('../myspace/input/taobao_round2_img_inst_80_id3000.json', 'w') as f:
        json.dump(img_sub_df.to_dict('records'), f)

    with open('../myspace/input/taobao_round2_vid_inst_80_id3000.json', 'w') as f:
        json.dump(vid_sub_df.to_dict('records'), f)


if __name__ == "__main__":
    main()