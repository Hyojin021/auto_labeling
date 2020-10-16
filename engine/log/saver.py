import shutil
import torch
import os
import glob
from datetime import datetime
from engine.log import summarise, logger

TODAY = datetime.today().strftime('%Y%m%d')


class Saver(object):
    """
        Log, Tensorboard, Checkpoint 저장을 위한 Code
        1번 실행 할 때마다 실행된 날짜를 기준으로 폴더가 생성되며
        해당 폴더 내부에는, Log파일, Checkpoint파일, Tensorboard파일이 생성되게 된다.
    """
    def __init__(self, config):
        self.config = config
        self.directory = os.path.join('run', config.projectname)
        self.today_runs = glob.glob(os.path.join(self.directory, f'{TODAY}_*'))
        self.runs = sorted([int(os.path.basename(f).split('_')[1]) for f in self.today_runs] if self.today_runs else [0])
        run_id = self.runs[-1] + 1

        self.experiment_dir = os.path.join(self.directory, f'{TODAY}_{run_id}')

        if not os.path.isdir(self.experiment_dir):
            os.makedirs(self.experiment_dir, exist_ok=True)

        self.summary = summarise.TensorboardSummary(self.experiment_dir)
        self.getlogger = logger.get_logger(self.experiment_dir)

    def save_checkpoint(self, state, is_best=None, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        if is_best is not None:
            best_f1_score = is_best
            with open(os.path.join(self.experiment_dir, 'best_f1_score.txt'), 'w', encoding='utf-8') as t:
                t.write(str(best_f1_score))
            if self.runs:
                previous_f1 = [.0]
                for run_id in self.runs:
                    path = os.path.join(self.directory, f'{TODAY}_{run_id}', 'best_f1_score.txt')
                    if os.path.exists(path):
                        with open(path, 'r', encoding='utf-8') as t:
                            f1_score = float(t.readline())
                            previous_f1.append(f1_score)
                max_f1_score = max(previous_f1)
                if best_f1_score > max_f1_score:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))