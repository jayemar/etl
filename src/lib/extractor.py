#!/usr/bin/env python

"""Handles the Extract portion of the E-T-L pipeline

Usage:
  extractor.py <ml_cfg>

Options:
  -h --help         Show this config
"""
from docopt import docopt

from etl import ETL
from utils import get_data_generator
from utils import handle_config


class Extractor(ETL):

    def retrieve_data(self, ml_cfg):
        self.ml_cfg = handle_config(ml_cfg)
        return get_data_generator('train', self.ml_cfg.get('batch_size'))

    def get_test_data(self):
        return get_data_generator('test', self.ml_cfg.get('batch_size'))


if __name__ == '__main__':
    args = docopt(__doc__)
    ml_cfg = args.get('<ml_cfg>')
    extractor = Extractor()

    gen = extractor.retrieve_data(ml_cfg)
    count = 0
    for data, meta in gen:
        count += 1
    batch_size = extractor.ml_cfg.get('batch_size')
    print("{} training batches of size {}".format(count, batch_size))

    gen = extractor.get_test_data()
    count = 0
    for data, meta in gen:
        count += 1
    print("{} test batches of size {}".format(count, batch_size))
