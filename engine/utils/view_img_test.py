import cv2
import os
import xml.etree.ElementTree as ET

def load_xml(anno_path):
    target = []
    tree = ET.parse(anno_path)
    root = tree.getroot()

    filename = root.find('filename').text

    for obj in root.findall('object'):
        name = obj.find('name').text

        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        target.append([xmin, ymin, xmax, ymax])
    return filename, target


img_dir = '/media/jsk/data/namwon/defect/Not_Print_img/img'
anno_dir = '/media/jsk/data/namwon/defect/Not_Print_img/anno'

anno_paths = [os.path.join(anno_dir, f.name) for f in os.scandir(anno_dir)]

for anno_path in anno_paths:
    filename, bboxes = load_xml(anno_path)
    img_paths = os.path.join(img_dir, filename)
    img = cv2.imread(img_paths, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 512, 512)
    cv2.imshow('img', img)
    cv2.waitKey(0)


