import numpy as np
import torch
import cv2
import random
from typing import List, Dict
from torch.utils.data.sampler import Sampler
from torchvision import transforms as T


def transform_tr():
    trasnform = T.Compose([Normalizer(),
                           Resize(),
                           ToTensor()])
    return trasnform

def transform_val():
    trasnform = T.Compose([Normalizer(),
                           Resize(),
                           ToTensor()])
    return trasnform


def collater(batch):

    imgs = [b['img'] for b in batch]
    annots = [b['annot'] for b in batch]
    scales = [b['scale'] for b in batch]

    widths = [int(i.shape[0]) for i in imgs]
    heights = [int(i.shape[1]) for i in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], : ] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class ToTensor(object):
    def __call__(self, sample):
        img = sample['img']
        annot = sample['annot']
        scale = sample['scale']

        img = torch.from_numpy(img)
        annot = torch.from_numpy(annot)

        return {'img': img, 'annot': annot, 'scale': scale}


class Resize(object):
    def __init__(self, resize: List[int]=[512, 512]):
        self.height = resize[0]
        self.weight = resize[1]

    def __call__(self, sample: Dict[str, int],
                 min_side=512, max_side=512):

        img, annots = sample['img'], sample['annot']

        h, w, c = img.shape

        smallest_side = min(h, w)

        # rescale the image so that the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio

        largest_side = max(h, w)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        img = cv2.resize(img, (int(round(h*scale)), int(round(w*scale))), cv2.INTER_LINEAR)
        h, w, c = img.shape

        pad_w = 32 - h % 32
        pad_h = 32 - w % 32

        new_img = np.zeros((h + pad_w, w + pad_h, c)).astype(np.float32)
        new_img[:h, :w, :] = img.astype(np.float32)
        annots[:, :4] *= scale

        return {'img': new_img, 'annot': annots, 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']

        return {'img': (img.astype(np.float32) - self.mean) / self.std, 'annot': annots}


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
