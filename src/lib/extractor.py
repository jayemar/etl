#!/usr/bin/env python

import pandas as pd
import pywt
from scipy import signal as dsp
import seaborn as sns
import yaml

from etl import ETL
from utils import get_data_generator
from utils import handle_config

sns.set()


"""
This file should handle the E-T-L portion to take data extracted from the
parquet files and translate it into something to feed to the ML
"""


class Extractor(ETL):

    def retrieve_data(self, ml_cfg):
        self.ml_cfg = handle_config(ml_cfg)

    def _get_data_gen(self, data_type):
        """Return a data generator for data type of either 'train' or 'test'"""
        train_gen = load_data(data_type, cfg.get('num_lines'))

    def _load_data(self, data_type, num_lines):
        train_gen = get_data_generator('train', cfg.get('num_lines'))
        train, meta_train = next(train_gen)
        test_gen = get_data_generator('test', cfg.get('num_lines'))
        test, meta_test = next(test_gen)



def wavelet_transform(signals, wavelet, level):
    cAs, cDs = list(), list()
    for phase in signal:
        cA, *cD = pywt.wavedec(phase, wavelet, level=level)
        cAs.append(cA)
        cDs.append(cD[0])
    return cAs, cDs


def sort_and_reindex_peaks(df):
    df = df.sort_values('peak_index', inplace=False)
    df = df.reset_index(inplace=False, drop=True)
    return df


def get_peak_info(cD):
    args = cfg.get('peak_finder_args')
    peaks, pinfo = dsp.find_peaks(cD, **args)
    pinfo.update({'peak_index': peaks})

    valleys, vinfo = dsp.find_peaks(-cD, **args)
    if 'peak_heights' in vinfo:
        vinfo.update({'peak_heights': -vinfo.get('peak_heights')})
    vinfo.update({'peak_index': valleys})
    
    return pd.concat([pd.DataFrame(pinfo), pd.DataFrame(vinfo)])


def within_ratio(h1, h2, ratio):
    """Check if h2 (height2) has the opposite sign of h1 (height1)
    and is smaller by an amount within the given ratio
    """
    resp = False
    if (h1 < 0 and h2 > 0) or (h1 > 0 and h2 < 0):
        if (((abs(h1) * (1 - ratio)) < abs(h2)) and ((abs(h1) * (1 + ratio)) > abs(h2))):
            resp = True
    return resp

def remove_symmetric_pulses(df, height_ratio, max_dist, train_len):
    working_df = df.copy()
    to_remove = set()
    skip_until = -1 
    train_count = 0
    for idx, row in df.iterrows():
        # If indexes were already added to to_remove then we don't want to check the pulses
        if idx < skip_until:
            continue
        # if the next peak is within max_dist...
        try:
            if (row.peak_index + max_dist) >= df.iloc[idx + 1].peak_index:
                # ...and if the height is within height_ratio...
                if within_ratio(row.peak_heights, df.iloc[idx + 1].peak_heights, height_ratio):
                    # ...remove the symmetric pulses and the pulse train
                    to_remove.update([idx, idx + 1])
                    h2_index = df.iloc[idx + 1].peak_index
                    train = df[df.peak_index.between(h2_index, h2_index + train_len)]
                    train_count += len(train)
                    skip_until = train.index.values[-1]
                    to_remove.update(train.index.values)
        except IndexError:
            # End of df
            break
    for i in to_remove:
        working_df.drop(index=i, inplace=True)
    return working_df
