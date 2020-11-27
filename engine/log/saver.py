import torch
import os
from datetime import datetime

TODAY = datetime.today().strftime('%Y%m%d')


class Saver(object):
    """
        Log, Tensorboard, Checkpoint 저장을 위한 Code
        1번 실행 할 때마다 실행된 날짜를 기준으로 폴더가 생성되며
        해당 폴더 내부에는, Log파일, Checkpoint파일, Tensorboard파일이 생성되게 된다.
    """
    def __init__(self, config):
        self.config = config
        self.directory = os.path.join('./engine/run', config.projectname)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)


    def save_checkpoint(self, state, is_best=None, filename='best_f1_score_model.pth.tar'):

        if is_best is not None:
            filename = os.path.join(self.directory, filename)
            torch.save(state, filename)