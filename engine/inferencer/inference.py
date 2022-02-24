from tabnanny import filename_only
import torch
import numpy as np
import os
import cv2
import argparse
from engine.utils.control_xml import save_xml, load_label_map
from engine.retinanet.post_process import *
from engine.retinanet.utils import BBoxTransform, ClipBoxes


def load_image(img_path):

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
    h, w, c = image.shape

    pad_w = 32 - h % 32
    pad_h = 32 - w % 32

    new_img = np.zeros((h + pad_w, w + pad_h, c)).astype(np.float32)
    new_img[:h, :w, :] = image.astype(np.float32)

    image = np.array(new_img).astype(np.float)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]

    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    image = torch.as_tensor(image, dtype=torch.float32)

    return image, scale


def entropy_sampling(scores):

    # 아무것도 detect 하지 못한경우 scores가 0으로 됨. -> 학습이 매우 안되었다는 반증임.
    if len(scores) == 0:
        return 99

    scores = scores.cpu().numpy()
    # idxs = np.where(scores > 0.5)

    # scroe의 원소중에서 0이거나 0에수렴할정도로 작은경우가 있음. 1e-05를 더해서 방지
    img_entropy = -np.sum(scores * np.log2(scores + 1e-05))

    return img_entropy

def detect_image(i, total_img_cnt, img_path, model, class_list, al, xml_dir=None):
    img_name = os.path.basename(img_path)
    filename, ext = os.path.splitext(img_name)
    label_map = load_label_map(class_list)
    image, scale = load_image(img_path)

    regress_boxes = BBoxTransform()
    clip_boxes = ClipBoxes()


    with torch.no_grad():

        if not torch.cuda.is_available():
            raise RuntimeError(f'cuda must be used')

        image = image.cuda()
        model = model.cuda()

        model.eval()

        classification, regression, anchors = model(image)


        scores, labels, boxes = post_process(image, classification, regression, anchors, regress_boxes, clip_boxes)


        if al:
            img_entropy = entropy_sampling(scores)
            return img_entropy
        else:
            idxs = np.where(scores.cpu() > 0.5)[0]
            # score가 0.5 이상이것이 없는경우
            if len(idxs) == 0:
                fail = 1
                return fail, print(f'({i}/{total_img_cnt}) {filename}:Not Detect Object')
            fail = 0
            bboxes = []
            labels_name = []
            for j in range(idxs.shape[0]):
                bbox = boxes[idxs[j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)

                key = int(labels[idxs[j]])
                label_name = label_map[key]
                bboxes.append([x1, y1, x2, y2])
                labels_name.append(label_name)
            save_xml_name = os.path.join(xml_dir, filename + '.xml')
            print(f'({i}/{total_img_cnt}) {filename}: create xml')
            save_xml(bboxes, labels_name, image.shape[0], image.shape[1], img_name, save_xml_name)
            return fail, print(f'({i}/{total_img_cnt}) {filename}: create xml')

def run(img_dir, xml_dir, signals):
    from engine.cfg.config import Config

    ann_name = [os.path.splitext(f.name)[0] for f in os.scandir(xml_dir)]
    img_name_dict = {os.path.splitext(f.name)[0]: os.path.splitext(f.name)[1] for f in os.scandir(img_dir)}
    img_names = list(set(img_name_dict.keys()).difference(set(ann_name)))

    config = Config()
    # 2. Best Model Checkpoint 로드하기
    model_path = f'./engine/run/{config.projectname}/best_f1_score_model.pth.tar'
    model = torch.load(model_path)

    # 3. 모델에 이미지 하나씩 넣어서 추론하고, entropy 구하기

    total_img_cnt = len(img_names)
    fail_cnt = 0
    for i, img_name in enumerate(img_names):
        progress = (i + 1) / total_img_cnt * 100

        ext = img_name_dict[img_name]
        img_path = os.path.join(img_dir, img_name + ext)

        fail, _ = detect_image(i+1, total_img_cnt, img_path, model, config.label_map_path, al=False, xml_dir=xml_dir)
        if fail == 1:
            fail_cnt += 1
    success_cnt = total_img_cnt - fail_cnt
    print(f'Success img: {success_cnt}, Fail img: {fail_cnt}')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Dir to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to label_map listing class names')
    parser.add_argument('--save_dir', help='Dir to directory save images')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list, parser.save_dir)
