import xml.etree.ElementTree as ET
import random
from math import floor
import numpy as np
import os
import cv2

class CropImgxml(object):
    def __init__(self, args):
        self.args = args
        self.annos_path = [os.path.join(self.args.anno_dir, x.name) for x in os.scandir(self.args.anno_dir)]
        self.classes = {}
        with open(args.label_map, 'r', encoding='utf-8') as t:
            all_class = t.read().splitlines()
            for i, each in enumerate(all_class):
                self.classes[each] = i
        self.classes2 = {}
        for k, v in self.classes.items():
            self.classes2[v] = k

    def partial_img(self, anno_path, save_root_dir=None):
        filename, anno = self.load_xml(anno_path)
        img_path = os.path.join(self.args.img_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_obj = np.array(anno).shape[0]
        for i in range(num_obj):
            obj = anno[i]

            if self.classes['SOOT'] == obj[4] or \
               self.classes['SPOTTED'] == obj[4] or \
               self.classes['SHADOW'] or \
               self.classes['unknown']:

                # BBOX를 기준으로 좌상당 CROP
                crop_img_1, img_box_1 = self.crop_img_1(img, obj)
                crop_boxes_1 = self.adj_anno_box(anno, img_box_1)

                # bbox를 기준으로 우상단 CROP
                crop_img_2, img_box_2 = self.crop_img_2(img, obj)
                crop_boxes_2 = self.adj_anno_box(anno, img_box_2)

                # bbox를 기준으로 좌하단 CROP
                crop_img_3, img_box_3 = self.crop_img_3(img, obj)
                crop_boxes_3 = self.adj_anno_box(anno, img_box_3)

                # bbox를 기준으로 우하단 CROP
                crop_img_4, img_box_4 = self.crop_img_4(img, obj)
                crop_boxes_4 = self.adj_anno_box(anno, img_box_4)

                if save_root_dir is not None:
                    if not os.path.exists(save_root_dir):
                        os.makedirs(save_root_dir)
                    if not os.path.exists(os.path.join(save_root_dir, 'img')):
                        os.makedirs(os.path.join(save_root_dir, 'img'))
                    if not os.path.exists(os.path.join(save_root_dir, 'anno')):
                        os.makedirs(os.path.join(save_root_dir, 'anno'))

                    rand = np.random.randint(0, 9999999, 4)
                    base_name = os.path.basename(anno_path)
                    xml_name, _ = os.path.splitext(base_name)

                    # 좌상단 저장
                    if crop_img_1.shape[0] == self.args.crop_H and crop_img_1.shape[1] == self.args.crop_W:
                        img_filename = str(rand[0]) + f'_{xml_name}.bmp'
                        xml_filename = str(rand[0]) + f'_{xml_name}.xml'
                        img_save_path = os.path.join(save_root_dir, 'img', img_filename)
                        xml_save_path = os.path.join(save_root_dir, 'anno', xml_filename)
                        cv2.imwrite(img_save_path, crop_img_1)
                        self.save_xml(crop_boxes_1, crop_img_1.shape[0], crop_img_1.shape[1], img_filename, xml_save_path)

                    # 우상단 저장
                    if crop_img_2.shape[0] == self.args.crop_H and crop_img_2.shape[1] == self.args.crop_W:
                        img_filename = str(rand[1]) + f'_{xml_name}.bmp'
                        xml_filename = str(rand[1]) + f'_{xml_name}.xml'
                        img_save_path = os.path.join(save_root_dir, 'img', img_filename)
                        xml_save_path = os.path.join(save_root_dir, 'anno', xml_filename)
                        cv2.imwrite(img_save_path, crop_img_2)
                        self.save_xml(crop_boxes_2, crop_img_2.shape[0], crop_img_2.shape[1], img_filename, xml_save_path)

                    # 좌하단 저장
                    if crop_img_3.shape[0] == self.args.crop_H and crop_img_3.shape[1] == self.args.crop_W:
                        img_filename = str(rand[2]) + f'_{xml_name}.bmp'
                        xml_filename = str(rand[2]) + f'_{xml_name}.xml'
                        img_save_path = os.path.join(save_root_dir, 'img', img_filename)
                        xml_save_path = os.path.join(save_root_dir, 'anno', xml_filename)
                        cv2.imwrite(img_save_path, crop_img_3)
                        self.save_xml(crop_boxes_3, crop_img_3.shape[0], crop_img_3.shape[1], img_filename, xml_save_path)

                    # 우하단 저장
                    if crop_img_4.shape[0] == self.args.crop_H and crop_img_4.shape[1] == self.args.crop_W:
                        img_filename = str(rand[3]) + f'_{xml_name}.bmp'
                        xml_filename = str(rand[3]) + f'_{xml_name}.xml'
                        img_save_path = os.path.join(save_root_dir, 'img', img_filename)
                        xml_save_path = os.path.join(save_root_dir, 'anno', xml_filename)
                        cv2.imwrite(img_save_path, crop_img_4)
                        self.save_xml(crop_boxes_4, crop_img_4.shape[0], crop_img_4.shape[1], img_filename, xml_save_path)
            else:
                crop_img_c, img_box_c = self.crop_img_center(img, obj)
                crop_boxes_c = self.adj_anno_box(anno, img_box_c)

                if save_root_dir is not None:
                    if not os.path.exists(save_root_dir):
                        os.makedirs(save_root_dir, exist_ok=True)
                        os.makedirs(os.path.join(save_root_dir, 'img'), exist_ok=True)
                        os.makedirs(os.path.join(save_root_dir, 'anno'), exist_ok=True)

                    rand = np.random.randint(0, 9999999, 1)
                    base_name = os.path.basename(anno_path)
                    xml_name, _ = os.path.splitext(base_name)

                    # Center 저장
                    if crop_img_c.shape[0] == self.args.crop_H and crop_img_c.shape[1] == self.args.crop_W:
                        img_filename = str(rand[0]) + f'_{xml_name}.bmp'
                        xml_filename = str(rand[0]) + f'_{xml_name}.xml'
                        img_save_path = os.path.join(save_root_dir, 'img', img_filename)
                        xml_save_path = os.path.join(save_root_dir, 'anno', xml_filename)
                        cv2.imwrite(img_save_path, crop_img_c)
                        self.save_xml(crop_boxes_c, crop_img_c.shape[0], crop_img_c.shape[1], img_filename, xml_save_path)

        return "성공"

    def load_xml(self, anno_path):
        print(anno_path)
        target = []
        tree = ET.parse(anno_path)
        root = tree.getroot()

        filename = root.find('filename').text

        for obj in root.findall('object'):
            name = obj.find('name').text
            id = self.classes[name]
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            target.append([xmin, ymin, xmax, ymax, id])
        return filename, target

    def iou(self, img_box, anno_box):
        im_xmin, im_ymin, im_xmax, im_ymax = img_box[0], img_box[1], img_box[2], img_box[3]
        an_xmin, an_ymin, an_xmax, an_ymax = anno_box[0], anno_box[1], anno_box[2], anno_box[3]
        inter_xmin, inter_ymin = max(im_xmin, an_xmin), max(im_ymin, an_ymin)
        inter_xmax, inter_ymax = min(im_xmax, an_xmax), min(im_ymax, an_ymax)

        anno_box_area = (an_xmax - an_xmin) * (an_ymax - an_ymin)
        intersection_area = np.maximum(inter_xmax - inter_xmin + 1, 0) * np.maximum(inter_ymax - inter_ymin + 1, 0)

        inter_box = [inter_xmin, inter_ymin, inter_xmax, inter_ymax]
        iou = intersection_area / anno_box_area

        return iou, inter_box

    def crop_img_1(self, img, obj):
        an_xmin, an_ymin, an_xmax, an_ymax, id = obj[0], obj[1], obj[2], obj[3], obj[4]
        cx, cy, w, h = self.edgebox2centerbox(an_xmin, an_ymin, an_xmax, an_ymax)

        # bbox를 기준으로 좌상단으로 CROP
        crop_img_cx = cx - (self.args.crop_W / 4)
        crop_img_cy = cy - (self.args.crop_H / 4)

        im_xmin, im_ymin, im_xmax, im_ymax = self.centerbox2edgebox(crop_img_cx, crop_img_cy,
                                                                    self.args.crop_W, self.args.crop_H)
        crop_img = img[im_ymin:im_ymax, im_xmin:im_xmax, :]

        img_box = [im_xmin, im_ymin, im_xmax, im_ymax]

        return crop_img, img_box

    def crop_img_2(self, img, obj):
        an_xmin, an_ymin, an_xmax, an_ymax, id = obj[0], obj[1], obj[2], obj[3], obj[4]
        cx, cy, w, h = self.edgebox2centerbox(an_xmin, an_ymin, an_xmax, an_ymax)

        # bbox를 기준으로 우상단으로 CROP
        crop_img_cx = cx + (self.args.crop_W / 4)
        crop_img_cy = cy - (self.args.crop_H / 4)

        im_xmin, im_ymin, im_xmax, im_ymax = self.centerbox2edgebox(crop_img_cx, crop_img_cy,
                                                                    self.args.crop_W, self.args.crop_H)
        crop_img = img[im_ymin:im_ymax, im_xmin:im_xmax, :]

        img_box = [im_xmin, im_ymin, im_xmax, im_ymax]

        return crop_img, img_box

    def crop_img_3(self, img, obj):
        an_xmin, an_ymin, an_xmax, an_ymax, id = obj[0], obj[1], obj[2], obj[3], obj[4]
        cx, cy, w, h = self.edgebox2centerbox(an_xmin, an_ymin, an_xmax, an_ymax)

        # bbox를 기준으로 좌하으로 CROP
        crop_img_cx = cx - (self.args.crop_W / 4)
        crop_img_cy = cy + (self.args.crop_H / 4)

        im_xmin, im_ymin, im_xmax, im_ymax = self.centerbox2edgebox(crop_img_cx, crop_img_cy,
                                                                    self.args.crop_W, self.args.crop_H)
        crop_img = img[im_ymin:im_ymax, im_xmin:im_xmax, :]
        img_box = [im_xmin, im_ymin, im_xmax, im_ymax]

        return crop_img, img_box

    def crop_img_4(self, img, obj):
        an_xmin, an_ymin, an_xmax, an_ymax, id = obj[0], obj[1], obj[2], obj[3], obj[4]
        cx, cy, w, h = self.edgebox2centerbox(an_xmin, an_ymin, an_xmax, an_ymax)

        # bbox를 기준으로 우하단으로 CROP
        crop_img_cx = cx + (self.args.crop_W / 4)
        crop_img_cy = cy + (self.args.crop_H / 4)

        im_xmin, im_ymin, im_xmax, im_ymax = self.centerbox2edgebox(crop_img_cx, crop_img_cy,
                                                                    self.args.crop_W, self.args.crop_H)
        crop_img = img[im_ymin:im_ymax, im_xmin:im_xmax, :]
        img_box = [im_xmin, im_ymin, im_xmax, im_ymax]

        return crop_img, img_box

    def crop_img_center(self, img, obj):
        an_xmin, an_ymin, an_xmax, an_ymax, id = obj[0], obj[1], obj[2], obj[3], obj[4]
        cx, cy, w, h = self.edgebox2centerbox(an_xmin, an_ymin, an_xmax, an_ymax)

        # bbox를 중심으로 CROP
        crop_img_cx = cx
        crop_img_cy = cy

        im_xmin, im_ymin, im_xmax, im_ymax = self.centerbox2edgebox(crop_img_cx, crop_img_cy,
                                                                    self.args.crop_W, self.args.crop_H)
        crop_img = img[im_ymin:im_ymax, im_xmin:im_xmax, :]
        img_box = [im_xmin, im_ymin, im_xmax, im_ymax]

        return crop_img, img_box

    def adj_anno_box(self, anno, img_box):
        crop_boxes = []
        for obj_list in anno:
            iou, inter_box = self.iou(img_box, obj_list)
            if iou >= 0.6:
                crop_box_xmin = inter_box[0] - img_box[0]
                crop_box_ymin = inter_box[1] - img_box[1]
                crop_box_xmax = inter_box[2] - img_box[0]
                crop_box_ymax = inter_box[3] - img_box[1]
                crop_box = [crop_box_xmin, crop_box_ymin, crop_box_xmax, crop_box_ymax, obj_list[4]]
                crop_boxes.append(crop_box)
            else:
                continue
        return crop_boxes

    def edgebox2centerbox(self, xmin, ymin, xmax, ymax):
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w/2
        cy = ymin + h/2
        return cx, cy, w, h

    def centerbox2edgebox(self, cx, cy, w, h):
        xmin = cx - w/2
        ymin = cy - h/2
        xmax = cx + w/2
        ymax = cy + h/2
        return floor(xmin), floor(ymin), floor(xmax), floor(ymax)

    def save_xml(self, bboxs, heigth, width, fname, savefilepath):
        annotation = ET.Element('annotation') # root
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
            id = int(bboxs[i][4])
            ET.SubElement(object, 'name').text = str(self.classes2[id])
            ET.SubElement(object, 'pose').text = 'Unspecified'
            ET.SubElement(object, 'truncated').text = str(0)
            ET.SubElement(object, 'difficult').text = str(0)
            bndbox = ET.SubElement(object, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(bboxs[i][0]))
            ET.SubElement(bndbox, 'ymin').text = str(int(bboxs[i][1]))
            ET.SubElement(bndbox, 'xmax').text = str(int(bboxs[i][2]))
            ET.SubElement(bndbox, 'ymax').text = str(int(bboxs[i][3]))

        self.indent(annotation)
        tree = ET.ElementTree(annotation)

        tree.write(savefilepath)
        return tree

    def indent(self, elem, level=0):
        i = "\n" + level * " "

        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + " "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('CropImg')
    parser.add_argument('--root-dir', type=str, default='/media/jsk/data/namwon/defect/Not_Print_img/')
    parser.add_argument('--img-dir', type=str, default='/media/jsk/data/namwon/defect/Not_Print_img/img')
    parser.add_argument('--anno-dir', type=str, default='/media/jsk/data/namwon/defect/Not_Print_img/anno')
    parser.add_argument('--label-map', type=str, default='/home/jsk/PycharmProjects/pytorch-retinanet/dataloader/dataset/label_map/defect.name')
    parser.add_argument('--crop-H', type=int, default=512)
    parser.add_argument('--crop-W', type=int, default=512)
    parser.add_argument('--save-root-dir', type=str, default='/media/jsk/data/namwon/defect/Not_Print_img/crop')

    args = parser.parse_args()
    crop_img_xml = CropImgxml(args)

    annos_path = [os.path.join(args.anno_dir, x.name) for x in os.scandir(args.anno_dir)]

    ########################################## 임시 코드 시작##########################################
    soot_list = []
    for anno_path in annos_path:
        filename, targets = crop_img_xml.load_xml(anno_path)
        for target in targets:
            id = target[4]
            if id == 0:
                soot_list.append(anno_path)
                break
    random.shuffle(soot_list)
    not_soot_list = list(set(annos_path).difference(set(soot_list)))
    random.shuffle(not_soot_list)

    soot_test_list = soot_list[:int(len(soot_list) * 0.2)]
    not_soot_test_list = not_soot_list[:int(len(not_soot_list) * 0.2)]

    annos_path = list(set(annos_path).difference(set(soot_test_list)).difference(set(not_soot_test_list)))

    print(annos_path, len(annos_path))

    with open(f'{args.root_dir}/soot_test_list.txt', 'w', encoding='utf-8') as t:
        for soot in soot_test_list:
            t.write(f'{soot}\n')

    with open(f'{args.root_dir}/not_soot_test_list.txt', 'w', encoding='utf-8') as t:
        for not_soot in not_soot_test_list:
            t.write(f'{not_soot}\n')
    ########################################## 임시 코드 끝##########################################

    for anno_path in annos_path:
        crop_img_xml.partial_img(anno_path, args.save_root_dir)
