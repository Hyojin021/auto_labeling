import os
from engine.train import Trainer
from engine.visualize_single_image import detect_image
import argparse
from engine.parse_config import ParseConfig
import torch
from engine.utils.voc2coco import get_label2id, get_annpaths, convert_xmls_to_cocojson


def active_learning(img_dir, annot_dir, signals):
    # signals.progress.emit(1, 'in progress')
    # signals.error.emit(100, 'SUCCESS')
    # signals.success.emit(100, 'SUCCESS')
    # config파일 Load
    parser = argparse.ArgumentParser('train/val main')
    parser.add_argument('--load-config', '-c', default='./engine/config.yaml')

    config = ParseConfig(parser).parse_args()

    print(config)

    # image, annot, labelmap 경로만 따로 빼두기(자주 사용하기 때문)
    # 엄군에게 받은 경로 넣기

    labels = './engine/dataloader/dataset/label_map/study_miniproject.name'


    # voc2coco 만들기
    # (생성된 annotations(json파일)은 추론이 끝나면 제거할 것
    save_coco_json = config.root_dir + '/annotations'
    label2id = get_label2id(labels_path=labels)
    ann_paths = get_annpaths(
        ann_dir=annot_dir
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=save_coco_json,
        extract_num_from_imgid=True
    )
    del label2id, ann_paths


    # 학습부 (추후 Active Learning할때 trainer.validation에서 mAP 추출해야함)
    trainer = Trainer(config)
    progress_range = range(trainer.config.start_epoch, trainer.config.epoch)
    progress_len = len(progress_range)
    signals.progress.emit(0, 'Start training')
    for epoch in range(trainer.config.start_epoch, trainer.config.epoch):
        trainer.train(epoch)
        trainer.validation(epoch)
        progress = (epoch + 1) / progress_len * 100
        signals.progress.emit(progress, 'epoch %d complete' % epoch)
    trainer.writer.close()


    # 추론부
    # 1. 사용자가 Labeling했던 data는 제외시키기(덮어쓰기가 되면 안되기 때문)
    ann_name = [os.path.splitext(f.name)[0] for f in os.scandir(annot_dir)]
    img_name = [os.path.splitext(f.name)[0] for f in os.scandir(img_dir)]
    img_name = list(set(img_name).difference(set(ann_name)))
    img_paths = [img_dir + f'{i}.jpg' for i in img_name]

    # 2. Best Model Checkpoint 로드하기
    model_path = f'./run/{config.projectname}/best_model.pth.tar'
    model = torch.load(model_path)

    # 3. 모델에 이미지 하나씩 넣어서 추론하고, xml파일로 저장하기
    for img_path in img_paths:
        detect_image(img_path, model, labels, annot_dir)
