import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from engine.dataloader import get_dataloader
from engine.retinanet import model
from engine.retinanet import coco_eval
from engine.log.saver import Saver
from tqdm import tqdm
from engine.retinanet import losses
from collections import deque
from engine.log import logger, summarise

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


class Trainer(object):
    def __init__(self, config, img_dir, coco_json):
        self.config = config

        # Define Saver
        self.saver = Saver(self.config)

        # Define Tensorboard
        if self.config.tensorboard:
            self.summary = summarise.TensorboardSummary(self.saver.directory)
            self.writer = self.summary.create_summary()

        # Define Logger
        self.getlogger = logger.get_logger(self.saver.directory)
        self.logger = self.getlogger

        # Define DataLoader
        self.train_loader, self.n_train_img,\
        self.val_set, self.val_loader, self.n_val_img, self.n_classes = get_dataloader(self.config, img_dir, coco_json)

        # Define Network
        if self.config.depth == 18:
            self.retinanet = model.resnet18(num_classes=80, pretrained=False)
        elif self.config.depth == 34:
            self.retinanet = model.resnet34(num_classes=80, pretrained=False)
        elif self.config.depth == 50:
            self.retinanet = model.resnet50(num_classes=80, pretrained=False)
            # self.retinanet = model.resnet50(num_classes=self.n_classes, pretrained=True)
        elif self.config.depth == 101:
            self.retinanet = model.resnet101(num_classes=80, pretrained=False)
        elif self.config.depth == 152:
            self.retinanet = model.resnet152(num_classes=80, pretrained=False)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        # Define resume
        self.best_f1_score = .0
        if self.config.resume is not None:
            checkpoint = torch.load(self.config.resume)
            self.retinanet.load_state_dict(checkpoint)

            from engine.retinanet.model import ClassificationModel
            self.retinanet.classificationModel = ClassificationModel(256, num_classes=self.n_classes)

        # Define Optimizer
        self.optimizer = optim.Adam(self.retinanet.parameters(), lr=self.config.lr)

        # Define lr_schduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        # Define loss
        self.loss = losses.FocalLoss()
        self.loss_hist = deque(maxlen=500)

        # Define cuda
        if torch.cuda.is_available():
            self.retinanet = torch.nn.DataParallel(self.retinanet).cuda()
        else:
            raise ValueError('=> Cuda is not available. Check cuda')

        # check model summary
        # summary(self.retinanet, (3, 512, 512))

    def train(self, epoch):
        self.retinanet.train()
        self.retinanet.module.freeze_bn()
        epoch_loss = []

        print(f'Num training images: {self.n_train_img}')

        with tqdm(self.train_loader) as tbar:
            for iter_num, data in enumerate(tbar):
                self.optimizer.zero_grad()

                img = data['img'].cuda().float()
                annot = data['annot'].cuda()

                classification, regression, anchors = self.retinanet(img)

                cls_loss, reg_loss = self.loss(classification, regression, anchors, annot)

                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                epoch_loss.append(float(loss))
                self.loss_hist.append(float(loss))

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.retinanet.parameters(), 0.1)
                self.optimizer.step()

                if self.config.tensorboard:
                    self.writer.add_scalar('Train_Loss/classification_loss',
                                           cls_loss,
                                           iter_num + epoch*(len(self.train_loader)))
                    self.writer.add_scalar('Train_Loss/regression_loss',
                                           reg_loss,
                                           iter_num + epoch*(len(self.train_loader)))
                    self.writer.add_scalar('Train_Loss/total_loss',
                                           np.mean(self.loss_hist),
                                           iter_num + epoch*(len(self.train_loader)))

                tbar.set_description(f'Epoch: {epoch} | '
                                     f'Cls loss: {cls_loss:1.5f} | '
                                     f'Reg loss: {reg_loss:1.5f} | '
                                     f'Running loss: {np.mean(self.loss_hist):1.5f}')
                del cls_loss, reg_loss
        self.scheduler.step(np.mean(epoch_loss))

    def validation(self, epoch):
        print('Evaluating dataset')
        stats = coco_eval.evaluate_coco(self.val_set, self.retinanet, self.saver.directory)

        if stats is None:
            return

        # stats: 0~11까지 12개의 값이 존재
        # 0: mAP / 1: map .5 / 2: map .75 / 3: ap small / 4: ap medium / 5: ap large/
        # 6: ar Det1 / 7: ar Det10 / 8: ar Det100 / 9: ar small / 10: ar medium / 11: ar large

        if self.config.tensorboard:
            self.writer.add_scalar('Precision/mAP', stats[0], epoch)
            self.writer.add_scalar('Precision/mAP@50IOU', stats[1], epoch)
            self.writer.add_scalar('Precision/mAP@75IOU', stats[2], epoch)
            self.writer.add_scalar('Precision/mAP(samll)', stats[3], epoch)
            self.writer.add_scalar('Precision/mAP(medium)', stats[4], epoch)
            self.writer.add_scalar('Precision/mAP(large)', stats[5], epoch)
            self.writer.add_scalar('Recall/AR@1', stats[6], epoch)
            self.writer.add_scalar('Recall/AR@10', stats[7], epoch)
            self.writer.add_scalar('Recall/AR@100', stats[8], epoch)
            self.writer.add_scalar('Recall/AR@100(small)', stats[9], epoch)
            self.writer.add_scalar('Recall/AR@100(medium)', stats[10], epoch)
            self.writer.add_scalar('Recall/AR@100(large)', stats[11], epoch)

        mAP, AR = stats[0], stats[8]
        f1_score = 2 * (mAP * AR) / (mAP + AR)

        if f1_score > self.best_f1_score:
            self.best_f1_score = f1_score
            self.saver.save_checkpoint(self.retinanet.module, f1_score)