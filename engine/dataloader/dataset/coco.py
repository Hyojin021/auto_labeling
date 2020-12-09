import numpy as np
import os
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import cv2


class CocoDataset(Dataset):
    '''Coco data Style'''

    def __init__(self, img_dir, annot_dir, set_name='val', transform=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.set_name = set_name
        self.transfrom = transform
        self.coco = COCO(os.path.join(annot_dir, f'instances_{set_name}.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        _img = self.load_image(idx)
        _annot = self.load_annotations(idx)
        sample = {'img': _img, 'annot': _annot}

        if self.transfrom is not None:
            sample = self.transfrom(sample)
        return sample

    def load_image(self, image_index):

        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.img_dir, image_info['file_name'])

        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).astype(np.float32)/255.0

        return img


    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations
        # # Get Ground Truth
        # ann_idx = self.coco.getAnnIds(imgIds=self.image_ids[idx], iscrowd=False)
        # coco_ann = self.coco.loadAnns(ann_idx)
        #
        # annotations = np.zeros((0, 5))
        #
        # # Bounding Box do not have annotation info
        # if len(coco_ann) == 0:
        #     return annotations
        #
        # # Parse Annotation info
        # for idx, ann in enumerate(coco_ann):
        #     annotation = np.zeros((1, 5))
        #     annotation[0, :4] = ann['bbox']
        #     annotation[0, 4] = self.coco_label_to_label(ann['category_id'])
        #     annotations = np.append(annotations, annotation, axis=0)
        # # transform [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]
        # annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        # annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        # return annotations

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        return self.labels

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]


if __name__ == '__main__':

    root_dir = '/media/jsk/data/namwon/defect/all_data/crop'
    dataset = CocoDataset(root_dir)
    dataset.__getitem__(0)
