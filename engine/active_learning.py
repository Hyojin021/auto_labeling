import os
from engine.trainer.train import Trainer
from engine.inferencer.inference import detect_image
import torch
from engine.utils.voc2coco import convert_annot
from .cfg.config import Config


class ActiveLearning(object):
    def __init__(self):
        super(ActiveLearning, self).__init__()
        self.label_map = './engine/dataloader/dataset/label_map/defect.name'
        # 뭘 상속 받아야한다고 했는데??f
        # signals.progress.emit(1, 'in progress')
        # signals.error.emit(100, 'SUCCESS')
        # signals.success.emit(100, 'SUCCESS')


    def run(self, img_dir, xml_dir, signals):

        # TODO: image, annot, labelmap 경로만 따로 빼두기(자주 사용하기 때문)

        # voc2coco 만들기
        # label_map을 사전에 ui에서 생성할 수 있도록 해야함.
        coco_dir = os.path.join(os.path.dirname(xml_dir), 'annotations')
        convert_annot(xml_dir, self.label_map, coco_dir)

        # 학습부 (추후 Active Learning할때 trainer.validation에서 mAP 추출해야함)
        config = Config()
        trainer = Trainer(config, img_dir, coco_dir)

        progress_range = range(trainer.config.start_epoch, trainer.config.epoch)
        progress_len = len(progress_range)
        signals.progress.emit(0, 'Start training')

        for epoch in range(trainer.config.start_epoch, trainer.config.epoch):
            trainer.train(epoch)
            trainer.validation(epoch)
            progress = (epoch + 1) / progress_len * 100
            signals.progress.emit(progress, f'epoch {epoch} complete')

        if config.tensorboard:
            trainer.writer.close()

        # 추론부
        # 1. 사용자가 Labeling했던 data는 제외시키기(덮어쓰기가 되면 안되기 때문)
        ann_name = [os.path.splitext(f.name)[0] for f in os.scandir(xml_dir)]
        img_name = [os.path.splitext(f.name)[0] for f in os.scandir(img_dir)]
        img_name = list(set(img_name).difference(set(ann_name)))
        img_paths = [img_dir + f'/{i}.bmp' for i in img_name]

        # 2. Best Model Checkpoint 로드하기
        model_path = f'./engine/run/{config.projectname}/best_f1_score_model.pth.tar'
        model = torch.load(model_path)

        # 3. 모델에 이미지 하나씩 넣어서 추론하고, entropy 구하기

        all_entropy = {}
        for i, img_path in enumerate(img_paths):
            progress = (i + 1) / len(img_paths) * 100
            entropy = detect_image(img_path, model, self.label_map, al=True)
            all_entropy[img_path] = entropy
            signals.progress.emit(progress, f'Inferencing image : {i} / {len(img_paths)} complete')
            if i+1 == 100: break
        all_entropy = sorted(all_entropy.items(), key=lambda x: x[1], reverse=True)
        img_list = list(dict(all_entropy[:100]).keys())
        signals.unconfirmed.emit(img_list)


