from engine.dataloader.dataset import coco
from engine.dataloader.custom_transform import collater, transform_tr, transform_val, AspectRatioBasedSampler
from torch.utils.data import DataLoader

def get_dataloader(config, img_dir, coco_dir):

    train_set = coco.CocoDataset(img_dir, coco_dir, set_name='train', transform=transform_tr())
    val_set = coco.CocoDataset(img_dir, coco_dir, set_name='val', transform=transform_val())

    n_train_img = train_set.__len__()
    n_val_img = val_set.__len__()

    sampler = AspectRatioBasedSampler(train_set, batch_size=config.batch_size, drop_last=False)
    train_loader = DataLoader(train_set, num_workers=config.num_worker, collate_fn=collater, batch_sampler=sampler, pin_memory=True)

    if val_set is not None:
        sampler_val = AspectRatioBasedSampler(val_set, batch_size=1, drop_last=False)
        val_loader = DataLoader(val_set, num_workers=config.num_worker, collate_fn=collater, batch_sampler=sampler_val, pin_memory=True)

    return train_loader, n_train_img, val_set, val_loader, n_val_img, train_set.num_classes()