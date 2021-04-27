#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
description: save validation output to file for use in diagnostics
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

import os
import json
import pickle

# For diagnostic output
from hashlib import sha256
from time import time
from socket import gethostname

import logging
# logging = logging.getLogger("buildLogger")

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def add_scores(dat, actual, pred, target):
    """ Add actual and predicted target to test data """

    dat = dat.copy()
    try:
       dat['pred_' + target] = pred
       dat['actual_' + target] = actual
    except Exception as err:
        logging.error('Unable to add pred and actuals to data')
        raise err

    return dat

def hash_time(time, digits=10):
    """ Returns the hash of time.time() """
    output = sha256(str(time).encode()).hexdigest()[:digits]
    return output

def create_output_dir(base_output_dir):

    ## Try creating a folder until successful
    while True:

        date_time = time()

        ## Create folder with random_name
        model_id = hash_time(date_time)
        output_dir = os.path.join(base_output_dir, model_id)

        try:
            payload_dir = os.path.join(output_dir, 'payload')
            os.makedirs(payload_dir)
            break
        except FileExistsError:
            next
        except Exception as err:
            logging.exception('Failure creating output directory')
            raise err

    return date_time, model_id, output_dir, payload_dir

def save_model_output(model, configs, enc_dict, metadata, test_data=None):
    """ Save model output for either validation model or final model """

    output_dir = configs['output']['model_package']['dir'] or 'output/model'

    date_time, model_id, output_dir, payload_dir = create_output_dir(output_dir)

    info = {
        'date_created': date_time,
        'host_created': gethostname()
    }

    info = {**info, **metadata}

    ## Save info
    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        json.dump(info, f)

    with open(os.path.join(payload_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(payload_dir, 'configs.json'), 'w') as f:
        json.dump(configs, f)

    with open(os.path.join(payload_dir, 'enc_dict.pkl'), 'wb') as f:
        pickle.dump(enc_dict, f)

    if test_data is not None:
        with open(os.path.join(payload_dir, 'test_data.pkl'), 'wb') as f:
            pickle.dump(test_data, f)

    return model_id

