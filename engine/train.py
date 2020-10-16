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
from collections import deque
from engine.parse_config import ParseConfig

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


class Trainer(object):
    def __init__(self, config):
        self.config = config

        # Define Saver, Tensorboard, logger
        self.saver = Saver(config)
        self.writer = self.saver.summary.create_summary()
        self.logger = self.saver.getlogger

        # Define DataLoader
        self.train_loader, self.n_train_img,\
        self.val_set, self.val_loader, self.n_val_img, self.n_classes = get_dataloader(config)

        # Define Network
        if self.config.depth == 18:
            self.retinanet = model.resnet18(num_classes=self.n_classes, pretrained=True)
        elif self.config.depth == 34:
            self.retinanet = model.resnet34(num_classes=self.n_classes, pretrained=True)
        elif self.config.depth == 50:
            self.retinanet = model.resnet50(num_classes=self.n_classes, pretrained=True)
        elif self.config.depth == 101:
            self.retinanet = model.resnet101(num_classes=self.n_classes, pretrained=True)
        elif self.config.depth == 152:
            self.retinanet = model.resnet152(num_classes=self.n_classes, pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        # Define Optimizer
        self.optimizer = optim.Adam(self.retinanet.parameters(), lr=self.config.lr)

        # Define lr_schduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        # TODO: Define Loss
        self.loss_hist = deque(maxlen=500)

        # TODO: Define Metric

        # Define cuda
        if config.cuda:
            if torch.cuda.is_available():
                self.retinanet = torch.nn.DataParallel(self.retinanet).cuda()
            else:
                raise ValueError('=> Cuda is not available. Check cuda')

        # Define Checkpoint
        self.best_f1_score = .0
        if config.resume is not None:
            if not os.path.exists(config.resume):
                raise FileNotFoundError(f'=> no checkpoint found at {config.resume}')
            checkpoint = torch.load(config.resume)

    def train(self, epoch):
        self.retinanet.train()

        if self.config.cuda:
            self.retinanet.module.freeze_bn()
        epoch_loss = []
        print(f'Num training images: {self.n_train_img}')
        with tqdm(self.train_loader) as tbar:
            for iter_num, data in enumerate(tbar):
                self.optimizer.zero_grad()

                if self.config.cuda:
                    img = data['img'].cuda().float()
                    annot = data['annot']
                    cls_loss, reg_loss = self.retinanet([img, annot])
                else:
                    img = data['img'].float()
                    annot = data['annot']
                    cls_loss, reg_loss = self.retinanet([img, annot])

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
        if self.config.data_style == 'coco':
            print('Evaluating dataset')
            best_f1_score = .0
            stats = coco_eval.evaluate_coco(self.val_set, self.retinanet, self.saver.experiment_dir)
            # stats: 0~11까지 12개의 값이 존재
            # 0: mAP / 1: map .5 / 2: map .75 / 3: ap small / 4: ap medium / 5: ap large/
            # 6: ar Det1 / 7: ar Det10 / 8: ar Det100 / 9: ar small / 10: ar medium / 11: ar large
            self.writer.add_scalar('Precision/mAP', stats[0], epoch)
            self.writer.add_scalar('Precision/mAP@50IOU', stats[1], epoch)
            self.writer.add_scalar('Precision/mAP@75IOU', stats[2], epoch)
            self.writer.add_scalar('Precision/mAP(samll)', stats[3], epoch)
            self.writer.add_scalar('Precision/mAP(medium)', stats[4], epoch)
            self.writer.add_scalar('Precision/mAP(large)', stats[5], epoch)
            self.writer.add_scalar('Recall/AR@1', stats[6], epoch)
            self.writer.add_scalar('Recall/AR@10', stats[7], epoch)
            self.writer.add_scalar('Recall/AR@10', stats[8], epoch)
            self.writer.add_scalar('Recall/AR@100(small)', stats[9], epoch)
            self.writer.add_scalar('Recall/AR@100(medium)', stats[10], epoch)
            self.writer.add_scalar('Recall/AR@100(large)', stats[11], epoch)

        # elif self.config.data_style == 'csv' and self.config.csv_val is not None:
        #     print('Evaluating dataset')
        #     mAP = csv_eval.evaluate(self.val_loader, self.retinanet)

        mAP, AR = stats[0], stats[8]
        f1_score = 2 * (mAP * AR) / (mAP + AR)

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            self.saver.save_checkpoint(self.retinanet.module, best_f1_score)


def _get_config():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--projectname', type=str, default='CJ_DEFECT', help='projectname')

    parser.add_argument('--data_style', default='coco', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--root-dir', default='/media/jsk/data/namwon/defect/Not_Print_img/crop/', help='Path to COCO directory')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--resize', type=int, default=[512, 512], help='Resize Image')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--cuda', default=True, help='if cuda is available, True')
    parser.add_argument('--resume', type=str, default=None, help='if checkpoint exist, write checkpoint path')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start Epoch')
    parser.add_argument('--epoch', type=int, default=100, help='End Epoch')
    parser.add_argument('--tensorboard', default=True, help='Use Tensorboard')

    config = parser.parse_args()

    return config

def main():
    parser = argparse.ArgumentParser('train/val main')
    parser.add_argument('--load-config', '-c', default='./config.yaml')

    config = ParseConfig(parser).parse_args()
    print(config)

    trainer = Trainer(config)

    for epoch in range(trainer.config.start_epoch, trainer.config.epoch):
        trainer.train(epoch)
        trainer.validation(epoch)
    trainer.writer.close()

if __name__ == '__main__':
    main()
