import numpy as np
import random
import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir: str = None) -> List[str]:
    # If use annotation paths list
    if ann_dir is not None:
        ann_paths = [os.path.join(ann_dir, f.name) for f in os.scandir(ann_dir)]
        return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    filename = annotation_root.findtext('filename')
    # if path == 'Unkown':
    #     filename = annotation_root.findtext('filename')
    # else:
    #     filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(''.join(re.findall(r'\d+', img_id)))

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')

    if not os.path.isdir(output_jsonpath):
        os.makedirs(output_jsonpath, exist_ok=True)

    num_file = len(annotation_paths)
    random.shuffle(annotation_paths)
    train_annotation_paths = annotation_paths[:int(num_file * 0.8)]
    val_annotation_paths = annotation_paths[int(num_file * 0.8):]


    for a_path in tqdm(train_annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(os.path.join(output_jsonpath, 'instances_train.json'), 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

    del output_json_dict
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    for a_path in tqdm(val_annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(os.path.join(output_jsonpath, 'instances_val.json'), 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def convert_annot(ann_dir, label_map_path, output_dir):

    label2id = get_label2id(labels_path=label_map_path)
    ann_paths = get_annpaths(
        ann_dir=ann_dir
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=output_dir,
        extract_num_from_imgid=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default='C:/Users/jsk/data/study_miniproject/annot',
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default='../dataloader/dataset/label_map/study_miniproject.name',
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='C:/Users/jsk/data/study_miniproject/annotations/',
                        help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    args = parser.parse_args()
    convert_annot(args)