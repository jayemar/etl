#!/usr/bin/env python

"""Handles the Translate portion of the E-T-L pipeline

Usage:
  translator.py <ml_cfg>

Options:
  -h --help         Show this config
"""
from docopt import docopt

from functools import partial
import pandas as pd
import pywt
from scipy import signal as dsp
import seaborn as sns

from etl import ETL
from utils import handle_config
from utils import sort_and_reindex
from utils import within_ratio

sns.set()


class Translator(ETL):

    def retrieve_data(self, ml_cfg):
        self.ml_cfg = handle_config(ml_cfg)
        raise NotImplementedError("retrieve_data not complete for Translator")

    def do_stuff(self, cD):
        resort = partial(sort_and_reindex, col_name='peak_info')

        dfs = [resort(self.get_peak_info(sig)) for sig in cD]
        self.plot_phases(dfs)

        max_height = self.ml_cfg.get('max_height')
        height_ratio = self.ml_cfg.get('max_height_ratio')
        max_dist = self.ml_cfg.get('max_distance')
        max_peaks = self.ml_cfg.get('max_ticks_removal')
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
        args = self.ml_cfg.get('peak_finder_args')
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
    args = docopt(__doc__)
    ml_cfg = args.get('<ml_cfg>')
    translator = Translator()

    gen = translator.retrieve_data(ml_cfg)
    count = 0
    for data, meta in gen:
        count += 1
    batch_size = translator.ml_cfg.get('batch_size')
    print("{} training batches of size {}".format(count, batch_size))

    gen = translator.get_test_data()
    count = 0
    for data, meta in gen:
        count += 1
    print("{} test batches of size {}".format(count, batch_size))
