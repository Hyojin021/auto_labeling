class Config(object):

    # Global Config
    projectname = 'udk_labeler'
    label_map_path = './engine/dataloader/dataset/label_map/udk_labeler.names'
    data_style = 'coco'
    tensorboard = False

    # Config DataLoader
    batch_size = 5
    start_epoch = 0
    epoch = 10
    num_worker = 4
    pin_memory = True

    # Config model
    model = 'retinanet'
    depth = 50
    resume = None
    lr = 0.00001
