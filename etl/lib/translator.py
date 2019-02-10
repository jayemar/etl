#!/usr/bin/env python

"""Handles the Translate portion of the E-T-L pipeline

Usage:
  translator.py <ml_cfg> [-r <train_count>] [-e <test_count>] [-v <validation_count>]

Options:
  -h --help         Show this config
  -r train_count        Upper limit on the number of training batches to pull
  -e test_count         Upper limit on the number of test batches to pull
  -v validation_count   Upper limit on the number of validation batches to pull
"""
from docopt import docopt

from functools import partial
from multiprocessing import Pool
import pandas as pd
import pywt
from scipy import signal as dsp

from .etl import ETL
from .extractor import Extractor
from .utils import etl_cli
from .utils import handle_config
from .utils import sort_and_reindex
from .utils import within_ratio


def translator_func(data_meta, cfg):
    """Template function to do work in pool"""
    data, meta = data_meta
    return data, meta

class Translator(ETL):

    def __init__(self, env_cfg={}):
        super(Translator, self).__init__(env_cfg)
        self.env_cfg = env_cfg

    def retrieve_data(self, ml_cfg):
        """Pass config file to retrieve generator for training data"""
        self.ml_cfg = ml_cfg
        self.func = partial(translator_func, cfg=ml_cfg)
        pool = Pool(self.env_cfg.get('translator_pool', 1))
        data_gen = self.data_in.retrieve_data(self.ml_cfg)
        for data, meta in pool.imap(self.func, data_gen):
            yield data, meta

    def get_test_data(self):
        """Retrieve generator for test data based on previous config"""
        pool = Pool(self.env_cfg.get('translator_pool', 1))
        data_gen = self.data_in.get_test_data()
        for data, meta in pool.imap(self.func, data_gen):
            yield data, meta

if __name__ == '__main__':
    args = docopt(__doc__)
    translator = Translator()
    extractor = Extractor()
    translator.set_data_input(extractor)
    etl_cli(translator, args)
