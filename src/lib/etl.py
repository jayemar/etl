#!/usr/bin/env python

"""Parent class/interface for ETL classes"""

import yaml

from lib.utils import handle_config


class ETL:
    def __init__(self, cfg=None, generator=None):
        self.cfg = handle_config(cfg)

