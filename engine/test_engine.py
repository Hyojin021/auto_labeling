import argparse
import unittest

from engine.parse_config import ParseConfig
from engine.trainer.train import Trainer


class TestEngine(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEngine, self).__init__(*args, **kwargs)
        parser = argparse.ArgumentParser('train/val main')
        parser.add_argument('--load-config', '-c', default='./config.yaml')

        config = ParseConfig(parser).parse_args()
        print(config)
        self.trainer = Trainer(config)

    def test_engine(self):
        for epoch in range(self.trainer.config.start_epoch, self.trainer.config.epoch):
            self.trainer.train(epoch)
            self.trainer.validation(epoch)
        self.trainer.writer.close()

if __name__ == '__main__':
    unittest.main()
