import torch
import numpy as np
import time
import os
import cv2
import argparse
import xml.etree.ElementTree as ET


def save_xml(bboxs, classes, heigth, width, fname, savefilepath):
    annotation = ET.Element('annotation')  # root
    ET.SubElement(annotation, 'folder').text = 'Unkown'
    ET.SubElement(annotation, 'filename').text = str(fname)
    ET.SubElement(annotation, 'path').text = 'Unkown'
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unkown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(heigth)
    ET.SubElement(size, 'depth').text = str(3)
    ET.SubElement(annotation, 'segmented').text = str(0)
    for i in range(len(bboxs)):
        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = str(classes[i])
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = str(0)
        ET.SubElement(object, 'difficult').text = str(0)
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(bboxs[i][0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(bboxs[i][1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(bboxs[i][2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(bboxs[i][3]))

    indent(annotation)
    tree = ET.ElementTree(annotation)

    tree.write(savefilepath)

    return tree


def indent(elem, level=0):
    i = "\n" + level * " "

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + " "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def entropy_sampling(scores):
    scores = scores.cpu().numpy()
    obj_entropy = -np.sum(scores * np.log(scores), axis=1)
    img_entropy = np.max(obj_entropy)

    return img_entropy

def detect_image(img_path, model, class_list, al):

    img_name = os.path.basename(img_path)
    filename, _ = os.path.splitext(img_name)
    with open(class_list, 'r', encoding='utf-8') as f:
        classes = f.read().split()

    labels = {}
    for i, cls in enumerate(classes):
        labels[i] = cls

    image = cv2.imread(img_path)
    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 512
    max_side = 512
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale

    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    image = np.array(image).astype(np.float)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]

    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    image = torch.as_tensor(image, dtype=torch.float32)

    with torch.no_grad():

        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()

        model.training = False
        model.eval()

        nms_scores, transformed_anchors = model(image)

        if al:
            img_entropy = entropy_sampling(nms_scores)
            return img_entropy
        else:
            idxs = (nms_scores > 0.5)
            scores = nms_scores[idxs]
            scores, classification = scores.max(dim=1)
            bboxes = []
            labels_name = []

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)

                key = int(classification[idxs[0][j]])
                label_name = labels[key]
                bboxes.append([x1, y1, x2, y2])
                labels_name.append(label_name)
            return bboxes, labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Dir to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to label_map listing class names')
    parser.add_argument('--save_dir', help='Dir to directory save images')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list, parser.save_dir)
