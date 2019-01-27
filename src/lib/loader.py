#!/usr/bin/env python

"""Handles the Loader portion of the E-T-L pipeline

Usage:
  loader.py <ml_cfg>

Options:
  -h --help         Show this config
"""
from docopt import docopt

from etl import ETL
from utils import handle_config


class Loader(ETL):

    def retrieve_data(self, ml_cfg):
        self.ml_cfg = handle_config(ml_cfg)


if __name__ == '__main__':
    args = docopt(__doc__)
    ml_cfg = args.get('<ml_cfg>')
    loader = Loader()

    gen = loader.retrieve_data(ml_cfg)
    count = 0
    for data, meta in gen:
        count += 1
    batch_size = loader.ml_cfg.get('batch_size')
    print("{} training batches of size {}".format(count, batch_size))

    gen = loader.get_test_data()
    count = 0
    for data, meta in gen:
        count += 1
    print("{} test batches of size {}".format(count, batch_size))
