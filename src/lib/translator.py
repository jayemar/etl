#!/usr/bin/env python

from functools import partial
import pandas as pd
import pywt
from scipy import signal as dsp
import seaborn as sns
import yaml

from etl import ETL
from model import Model
from utils import handle_config
from utils import sort_and_reindex
from utils import within_ratio

sns.set()


"""
This file should handle the E-T-L portion to take data extracted from the
parquet files and translate it into something to feed to the ML
"""


class Translator(ETL):

    def retrieve_data(self, ml_cfg):
        self.ml_cfg = handle_config(ml_cfg)
        raise NotImplementedError("retrieve_data not complete for Translator")

    def do_stuff(self, cD):
        resort = partial(sort_and_reindex, col_name='peak_info')

        dfs = [resort(self.get_peak_info(sig)) for sig in cD]
        self.plot_phases(dfs)

        max_height = cfg.get('max_height')
        height_ratio = cfg.get('max_height_ratio')
        max_dist = cfg.get('max_distance')
        max_peaks = cfg.get('max_ticks_removal')
        dfs = [
            self.remove_symmetric_pulses(df, height_ratio, max_dist, max_peaks)
            for df in dfs
        ]
        dfs = [df[abs(df.peak_heights) < max_height] for df in dfs]
        dfs = [resort(df) for df in dfs]
        self.plot_phases(dfs)

    def wavelet_transform(self, signals, wavelet, level):
        cAs, cDs = list(), list()
        for sig in signals:
            cA, *cD = pywt.wavedec(sig, wavelet, level=level)
            cAs.append(cA)
            cDs.append(cD[0])
        return cAs, cDs

    def sort_and_reindex_peaks(self, df):
        df = df.sort_values('peak_index', inplace=False)
        df = df.reset_index(inplace=False, drop=True)
        return df

    def get_peak_info(self, cD):
        args = cfg.get('peak_finder_args')
        peaks, pinfo = dsp.find_peaks(cD, **args)
        pinfo.update({'peak_index': peaks})

        valleys, vinfo = dsp.find_peaks(-cD, **args)
        if 'peak_heights' in vinfo:
            vinfo.update({'peak_heights': -vinfo.get('peak_heights')})
        vinfo.update({'peak_index': valleys})

        return pd.concat([pd.DataFrame(pinfo), pd.DataFrame(vinfo)])

    def remove_symmetric_pulses(self, df, height_ratio, max_dist, train_len):
        working_df = df.copy()
        to_remove = set()
        skip_until = -1
        train_count = 0
        for idx, row in df.iterrows():
            # If indexes were already added to to_remove then we don't
            # want to check the pulses
            if idx < skip_until:
                continue
            # if the next peak is within max_dist...
            try:
                if (row.peak_index + max_dist) >= df.iloc[idx + 1].peak_index:
                    # ...and if the height is within height_ratio...
                    if within_ratio(row.peak_heights,
                                    df.iloc[idx + 1].peak_heights,
                                    height_ratio):
                        # ...remove the symmetric pulses and the pulse train
                        to_remove.update([idx, idx + 1])
                        h2_index = df.iloc[idx + 1].peak_index
                        train = df[df.peak_index.between(h2_index,
                                                         h2_index + train_len)]
                        train_count += len(train)
                        skip_until = train.index.values[-1]
                        to_remove.update(train.index.values)
            except IndexError:
                # End of df
                break
        for i in to_remove:
            working_df.drop(index=i, inplace=True)
        return working_df


if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        cfg = yaml.load(f)

    DATA_DIR = '../data'
    test_start = 8712  # TODO: What is this?

    model = Model()
    print("Complete")
