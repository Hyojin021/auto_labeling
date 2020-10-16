import pandas as pd
import os
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict


def iou(pred_box, true_box):

    p_xmin, p_ymin, p_xmax, p_ymax = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
    t_xmin, t_ymin, t_xmax, t_ymax = true_box[0], true_box[1], true_box[2], true_box[3]

    inter_xmin, inter_ymin = max(p_xmin, t_xmin), max(p_ymin, t_ymin)
    inter_xmax, inter_ymax = min(p_xmax, t_xmax), min(p_ymax, t_ymax)

    inter_area = np.maximum(inter_xmax - inter_xmin + 1, 0) * np.maximum(inter_ymax - inter_ymin + 1, 0)
    pred_area = (p_xmax - p_xmin + 1) * (p_ymax - p_ymin + 1)
    true_area = (t_xmax - t_xmin + 1) * (t_ymax - t_ymin + 1)
    union_area = true_area + pred_area - inter_area

    iou = inter_area / union_area

    return iou


def load_xml(anno_path):

    tree = ET.parse(anno_path)
    root = tree.getroot()
    target = defaultdict(list)

    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        target[name].append([xmin, ymin, xmax, ymax])

    return target


def confusion_metric(true_xml_dir, pred_xml_path, classes=['NG', 'OK']):
    _metric = pd.DataFrame(0, index=classes, columns=classes)
    xml_name = os.path.basename(pred_xml_path)
    true_xml_path = os.path.join(true_xml_dir, xml_name)

    p = load_xml(pred_xml_path)
    print(pred_xml_path)
    if p.get('SOOT') is None:
        if not os.path.exists(true_xml_path):
            _metric.iloc[1, 1] += 1 # 정상
        else:
            t = load_xml(true_xml_path)
            if t.get('SOOT') is None:
                _metric.iloc[1, 1] += 1 # 정상
            else:
                _metric.iloc[0, 1] += 1 # 미검(2종과오)
    else:
        if not os.path.exists(true_xml_path):
            _metric.iloc[1, 0] += 1 # 과검(1종과오)
        else:
            t = load_xml(true_xml_path)

            if t.get('SOOT') is None:
                _metric.iloc[1, 0] += 1  # 과검(1종과오)
            else:
                p_bboxes = p.get('SOOT')
                t_bboxes = t.get('SOOT')

                ious = []
                for t_bbox in t_bboxes:
                    for p_bbox in p_bboxes:
                        ious.append(iou(p_bbox, t_bbox))
                cnt = 0
                for i in ious:
                    if i >= 0.5:
                        _metric.iloc[0, 0] += 1 # 불
                        break
                    else:
                        cnt += 1
                        if cnt == len(ious):
                            _metric.iloc[0, 1] += 1  # 미검(2종과오)

    return _metric



class ConfusionMetric:
    def __init__(self, classes=['NG', 'OK']):
        self._classes = classes
        self._metric = pd.DataFrame(0, index=self._classes, columns=self._classes)

    def reset(self):
        for col in self._metric.colums:
            self._metric[col].values[:] = 0

    def update(self, value):
        self._metric += value

    def result(self):
        return self._metric


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--true_xml_dir', help='Path to directory containing images')
    parser.add_argument('--pred_xml_dir', help='Path to model')
    args = parser.parse_args()

    pred_xml_paths = [os.path.join(args.pred_xml_dir, x.name) for x in os.scandir(args.pred_xml_dir)]

    c_metric = ConfusionMetric()
    for pred_xml_path in pred_xml_paths:
        value = confusion_metric(args.true_xml_dir, pred_xml_path)
        c_metric.update(value)
        print(c_metric.result())
