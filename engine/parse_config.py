import os
import argparse
import yaml


class ParseConfig(object):

    def __init__(self, parser):
        self.parser = parser
        self.args = self.parser.parse_args()

    def parse_args(self):
        _, ext = os.path.splitext(self.args.load_config)

        if not self.args.load_config:
            raise FileNotFoundError("Can not found config file")

        config = self.parser.parse_args(namespace=self._create_dict())

        return config

    def _create_dict(self):
        with open(self.args.load_config, 'r', encoding='utf-8') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(yaml.load(f, Loader=yaml.SafeLoader))
        return t_args