import logging
from logging import handlers
import os


def get_logger(save_dir):
    mylogger = logging.getLogger('CJ PROJECT - defect')
    mylogger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s, %(message)s')

    loghandler = handlers.TimedRotatingFileHandler(filename=os.path.join(save_dir, 'logging.log'),
                                                   when='midnight',
                                                   interval=1,
                                                   encoding='utf-8')
    loghandler.setFormatter(formatter)
    loghandler.suffix = '%Y%m%d'
    mylogger.addHandler(loghandler)

    return mylogger