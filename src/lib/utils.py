#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from pyarrow import parquet as pq
import yaml

DATA_DIR = "../data/"
TEST_START = 8712  # First 8711 lines are training data


def handle_config(config):
    """Allow for th use of a file name or dict"""
    if type(config) == str:
        with open(config, 'r') as f:
            config = yaml.load(f)
    return config
        

def get_data_generator(data_type, num_lines_per_batch):
    """data_type should be either 'train' or 'test'"""
    if data_type not in ['train', 'test']:
        raise ValueError("'data_type' must be one of 'train'/'test'")
    start = 0 if data_type == 'train' else 8712
    end = start + num_lines_per_batch
    meta_file = "metadata_{}.csv".format(data_type)
    meta_df = pd.read_csv(Path(DATA_DIR, meta_file))
    while True:
        data = pq.read_pandas(
            Path(DATA_DIR, data_type + '.parquet'),
            columns=[str(i) for i in range(start, end)]
        ).to_pandas().values.T
        meta = meta_df.iloc[start: end]
        start = end
        yield data, meta


def sort_and_reindex(df, col_name):
    df = df.sort_values(col_name, inplace=False)
    df = df.reset_index(inplace=False, drop=True)
    return df


def too_tall(df, threshold, vals=False):
    """Return index of peaks above threshold value
    Include height of peaks if vals == True
    """
    field = 'peak_heights'
    return_fields = ['peak_index', field] if vals else ['peak_index']
    return df[(abs(df[field])) > threshold][return_fields]


def within_ratio(h1, h2, ratio):
    """Check if h2 (height2) has the opposite sign of h1 (height1)
    and is smaller by an amount within the given ratio
    """
    resp = False
    if (h1 < 0 and h2 > 0) or (h1 > 0 and h2 < 0):
        if (((abs(h1) * (1 - ratio)) < abs(h2))
                and ((abs(h1) * (1 + ratio)) > abs(h2))):
            resp = True
    return resp


def pos_neg_pairs(df, threshold, hdiff, vdiff, vals=False):
    """Return index of peaks with a matching peak of the opposite polarity
    Differnce in peak heights must be within hdiff
    Differnce in sample index must be within vdiff
    Include height of peaks if vals == True
    """
    field = 'peak_heights'
    return_fields = ['peak_index', field] if vals else ['peak_index']
    tmp = df.sort_values('peak_index')[['peak_index', 'peak_heights']]
