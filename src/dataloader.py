#!/usr/bin/env python

"""Creates a DataLoader instance for retrieving train/test/validation data

Usage:
  dataloader.py <ml_cfg>

Options:
  -h --help         Show this config
"""
from docopt import docopt

from libs.etl import ETL
from libs.extrator import Extractor
from libs.loader import Loader
from libs.translator import Translator
from libs.utils import handle_config


class DataLoader(ETL):

    def __init__(self, env_cfg=None):
        super(DataLoader, self).__init__(env_cfg)
        self.extractor = Extractor(env_cfg)
        self.translator = Translator(env_cfg)
        self.loader = Loader(env_cfg)

    def retrieve_data(self, ml_cfg):
        self.ml_cfg = handle_config(ml_cfg)


if __name__ == '__main__':
    args = docopt(__doc__)
    ml_cfg = args.get('<ml_cfg>')
    dataloader = DataLoader()

    gen = dataloader.retrieve_data(ml_cfg)
    count = 0
    for data, meta in gen:
        count += 1
    batch_size = dataloader.ml_cfg.get('batch_size')
    print("{} training batches of size {}".format(count, batch_size))

    gen = dataloader.get_test_data()
    count = 0
    for data, meta in gen:
        count += 1
    print("{} test batches of size {}".format(count, batch_size))
