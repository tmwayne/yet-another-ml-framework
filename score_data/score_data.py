#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
description: build modeling pipeline after extracting data
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

import os
import json
import pickle
from time import time, gmtime, strftime, sleep

import logging
# logging = logging.getLogger("scoreLogger")

from modules..src.extract_model_data import (
    extract_data,
    read_query,
    run_query,
    run_copy_query,
    save_to_s3,
    get_data
)
from modules.src.helper_functions import (
    update_configs,
    inventory_models
)
from modules.src.build_model_pipeline import (
    extract_targets,
    check_missing_vals,
    encode_cols
)

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def get_remaining_targets(redshift_configs, scoring_table,
    targets, query=None):

    if not query:
        query = 'select distinct target from {}'.format(scoring_table)

    remaining_targets = list(get_data(redshift_configs, query).target)

    return [x for x in targets if x not in remaining_targets]

def find_latest_model(model_inventory, target, mtype='final'):
    """ Go through the model inventory and find for the target
    the model with the most recent time stamp """
    latest_date = 0
    latest_model = ''
    for pkg_id, info in model_inventory.items():
        if info['target'] == target and info['type'] == mtype:
            if info['date_created'] > latest_date:
                latest_date = info['date_created']
                latest_model = pkg_id

    return latest_model

def load_model_pkg(model_id, output_dir='output/model'):
    """ Load the contents of the model package into dict """
    payload_dir = os.path.join(output_dir, model_id, 'payload')

    output = {}
    for f in os.listdir(payload_dir):
        filename, ext = os.path.splitext(f)
        path = os.path.join(payload_dir, f)
        if ext == '.json':
            with open(path, 'r') as f:
                output[filename] = json.load(f)
        elif ext == '.pkl':
            with open(path, 'rb') as f:
                output[filename] = pickle.load(f)

    return output

def build_scoring_output(dat, snapshot_date, cust_id, target, preds):
    """ Build the dataframe that is to be export to Redshift """

    output = dat[[snapshot_date, cust_id]].copy()
    output['target'] = target
    output['prediction'] = preds
    output['date_scored'] = strftime('%Y%m%d', gmtime(time()))

    ## Order the columns before we return
    col_order = ['date_scored', snapshot_date, cust_id, 'target', 'prediction']
    return output[col_order]

def score_target(dat, target, model_inventory, configs):
    """
    Load latest model package for the target,
    encode the data using the data encoder in the package,
    score the data, and save it to Redshift

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

    ## LOAD MODEL
    ##############################
    logging.info('Loading package of latest model...')
    try:
        model_id = find_latest_model(model_inventory, target, mtype='tuning')
        model_pkg = load_model_pkg(model_id,
            output_dir=configs['output']['model_package']['dir'])
        model_configs = model_pkg['configs']['model']
    except FileNotFoundError as e:
        logging.exception('Not package for target {}'.format(target))
        raise e

    ## ENCODE COLUMNS
    ##############################
    ## TODO: Fix one-hot encoding so that
    ## it can deal with unseen values
    logging.info('Encoding categorical columns...')
    dat_enc, enc_dict = encode_cols(
        dat,
        do_ohe=model_configs['preprocessing']['do_onehotencoding'],
        enc_dict=model_pkg['enc_dict']
    )

    ## SCORE CUSTOMERS
    ##############################
    logging.info('Scoring data...')
    preds = model_pkg['model'].predict_proba(
        dat_enc[model_configs['model_cols']]
    )[:,1]

    ## SAVE OUTPUT TO REDSHIFT
    ##############################
    logging.info('Building scoring output...')
    ## Build scoring output
    try:
        output = build_scoring_output(
            dat,
            model_pkg['configs']['data']['date'],
            model_pkg['configs']['data']['cust_id'],
            target,
            preds
        )

        logging.info('Saving scores to S3...')
        save_to_s3(
            output,
            configs['sql']['copy_query']['filename'],
            configs['sql']['s3']['bucket']
        )

        ## Copy ouput to redshift
        logging.info('Copying output from S3 to Redshift...')
        run_copy_query(
            configs['sql']['copy_query'],
            configs['sql']['s3'],
            configs['sql']['redshift']
        )
        sleep(10)
    except Exception as e:
        logging.exception('Failure saving output to Redshift.')
        raise e


def score_data(configs):
    """
    Score all new data with all models by retrieving latest
    modeling package for each target. Finally save scores to Redshift

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

    ## SET CONFIGURATIONS
    ##############################
    configs = update_configs(configs, configs['config_files'].values())

    ## TRUNCATE PROCESS FILES
    ##############################
    ## If a target fails to fit, we skip it
    ## and save the target to a file
    ## At the beginning of running this
    ## script, we clear the file
    if os.path.exists(configs['output']['failed_targets']['score']):
        os.remove(configs['output']['failed_targets']['score'])

    ## LOAD SCORING DATA
    ##############################
    logging.info('Extracting model data from Redshift...')
    try:
        dat = extract_data(
            configs['sql']['redshift'],
            configs['sql']['tables']['scoring']
        )
        logging.info('Successfully extracted data of size {}...'.format(dat.shape))
    except Exception as e:
        logging.exception('Failure extracting scoring data from Redshift')
        raise e

    ## IMPUTATION
    ##############################
    # Therefore the only imputation this script has
    # is to check for missing values, and upon
    # finding any raise a warning and fill with 0.
    logging.info('Checking for and imputing missing values...')
    dat = check_missing_vals(dat)

    ## LOAD MODEL INFO & TARGETS
    ##############################
    model_inventory = inventory_models(configs['output']['model_package']['dir'])
    targets = extract_targets(dat,
        ignore=configs['data']['global_target'])

    ## SCORE CUSTOMERS WITH EACH MODEL
    ##############################
    for n, target in enumerate(targets):

        logging.info('Using target {} of {}: {}'.format(
            n, len(targets), target))

        try:
            score_target(dat, target, model_inventory, configs)
        except:
            logging.exception('Failure scoring target {}'.format(target))
            ## If target fails in any way then we save it to a file
            with open(configs['output']['failed_targets']['score'], 'a') as f:
                f.write(target + '\n')
            continue

    ## CALC PROP THRESHOLDS
    ##############################
    logging.info('Calculate prediction thresholds...')
    try:
        query = read_query(configs['sql']['pred_thresholds'])
        run_query(configs['sql']['redshift'], query)
    except Exception as e:
        logging.exception('Failure calculating threshold.')
        raise e

# --------------------------------------------------------------------
# Default Configurations
# --------------------------------------------------------------------

CONFIGS = {
    'working_dir': '/path/to/project/root',
    'config_files': {
        'sql': 'configs/configs_sql.json',
        'output': 'configs/configs_output.json',
        'data': 'configs/configs_data.json'
    }
}

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == '__main__':

   score_data(CONFIGS)
