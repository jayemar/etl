#!/usr/bin/env python

"""Handles the Loader portion of the E-T-L pipeline

Usage:
  loader.py <ml_cfg> [-r <train_count>] [-e <test_count>] [-v <validation_count>]

Options:
  -h --help         Show this config
  -r train_count        Upper limit on the number of training batches to pull
  -e test_count         Upper limit on the number of test batches to pull
  -v validation_count   Upper limit on the number of validation batches to pull
"""
from docopt import docopt

from etl import ETL
from utils import etl_cli
from utils import handle_config


class Loader(ETL):

    def retrieve_data(self, ml_cfg):
        self.ml_cfg = handle_config(ml_cfg)
        return self.data_in.retrieve_data(ml_cfg)


if __name__ == '__main__':
    args = docopt(__doc__)
    loader = Loader()
    etl_cli(loader, args)
