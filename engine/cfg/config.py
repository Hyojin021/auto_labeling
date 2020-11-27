class Config(object):

    # Global Config
    projectname = 'udk_labeler'
    data_style = 'coco'
    tensorboard = False

    # Config DataLoader
    batch_size = 4
    start_epoch = 0
    epoch = 1
    num_worker = 4
    pin_memory = True

    # Config model
    model = 'retinanet'
    depth = 50
    resume = None
    lr = 0.00001
