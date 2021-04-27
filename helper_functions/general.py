#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
description: general helper functions for main.py
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

import os
from time import time

from functools import wraps
from inspect import getsource

import json
import yaml

from logging.config import dictConfig

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def print_func(func):
    print(getsource(func))

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print('Run time of {}: {:.2f} seconds'.format(
            func.__name__, (end_time - start_time)))
        return result
    return wrapper

def setup_logging(file):
    with open(file, 'r') as f:
        config = yaml.safe_load(f.read())
        dictConfig(config)

def load_config(file, *args, **kwargs):
    """ Load file based on file type """

    filename, ext = os.path.splitext(file)
    with open(file, 'r') as f:
        if ext == '.json':
            output = json.load(f, *args, **kwargs)
        elif ext == '.yaml':
            output = yaml.load(f, Loader=yaml.Loader, *args, **kwargs)
        else:
            raise Exception('File type {} not recognized'.format(ext))

        return output

def update_configs(configs, config_files):
    """
    Load configs from list of files and append to master config
    """
    for file in config_files:
        try:
            _configs = load_config(file)
            configs = {**configs, **_configs}
        except Exception as e:
            print('Unable to load {}'.format(file))
            raise e

    return configs

