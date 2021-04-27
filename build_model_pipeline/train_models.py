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
import logging
# logging = logging.getLogger("buildLogger")

from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from modules.src.extract_model_data import (
    extract_data,
    extract_names,
    run_from_template
)
from modules.src.helper_functions import (
    print_func,
    timeit,
    load_config,
    update_configs
)
from modules.src.build_model_pipeline import (
    check_missing_vals,
    encode_cols,
    fit_model,
    add_scores,
    save_model_output
)

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def mk_model_dict(training_configs):
    """ Load the libraries used for the models
    and create the model dictionary with the model classes """

    model_dict = {}
    for name, setup in training_configs.items():
        if setup['do_fit']:
            exec(setup['import_command'])
            model_dict[name] = eval(setup['model_class'])

    return model_dict

def rm_incomplete_weeks(dat, date_col, horizon, gap):
    """ Remove weeks where the target variable
    doesn't have complete information """
    dates = sorted(dat[date_col].unique())[:-(horizon + gap)]
    ind = dat[date_col].isin(dates)
    return dat.loc[ind,]

def extract_targets(dat, target='target', ignore=None):
    """ Extract columns from data with target in the name """
    return [col for col in dat.columns
        if target in col and ignore not in col]

def mk_info(type, algo, version, target, **kwargs):
    """ Make metadata dictionary for model package """
    metadata = {
        'type': type,
        'algo': algo,
        'version': version,
        'target': target
    }
    output = {**metadata, **dict(**kwargs)}
    return output

def train_target(dat, target, model_dict, enc_dict, configs, final_model=True):
    """
    For a specified target, we perform model selection
    and hyperparameter tuning, save the tuning model,
    fit a final model, and finally save that.

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

    configs = configs.copy()

    ## SPLIT TRAIN & TEST
    ##############################
    logging.info('Splitting training and test set...')
    x_train, x_test, y_train, y_test = train_test_split(
        dat.drop(labels=target, axis=1),
        dat[target],
        test_size=configs['model']['preprocessing']['test_size'],
        random_state=configs['model']['random_seed']
    )

    ## SPECIFY NON-MODELING COLS
    ##############################
    remove_cols = [x for x in configs['data']['targets'] if x != target]
    remove_cols.append(configs['data']['cust_id'])
    remove_cols.append(configs['data']['date'])
    remove_cols.append(configs['data']['global_target'])
    configs['data']['remove_cols'] = remove_cols

    ## MODEL SELECTION
    ##############################
    logging.info('Starting modeling selection and hyper-param tuning...')
    try:
        tuning_model, results, model_cols = fit_model(
            x=x_train, y=y_train,
            model=model_dict,
            data_configs=configs['data'],
            preprocessing_configs=configs['model']['preprocessing'],
            time_sampling_configs=configs['model']['time_sampling'],
            model_configs=configs['model'],
            training_configs=configs['training'],
            model_type='tuning'
        )
    except Exception as e:
        logging.exception('Failure training tuning model. Skipping target...')
        raise e

    ## Update configs
    configs['model']['model_cols'] = model_cols
    configs['model']['tuning_results'] = results

    ## VALIDATION
    ##############################
    ## Initial AUC score for validation
    logging.info('Evaluating test set...')
    pred_test = tuning_model.predict_proba(x_test[model_cols])[:,1]
    test_auc = roc_auc_score(y_test, pred_test)
    logging.info('AUC of test set is {:.2f}'.format(test_auc))

    ## SAVE FOR DIAGNOSTICS
    ##############################
    model_info = mk_info(
        type='tuning',
        algo=results['model'],
        version=0.1,
        target=target,
        auc=test_auc
    )
    test_data = add_scores(
        x_test[[configs['data']['cust_id'], configs['data']['date']]],
        y_test,
        pred_test,
        target
    )

    ## Save to disk
    logging.info('Saving tuning model to disk...')
    model_id = save_model_output(
        model=tuning_model,
        configs=configs,
        enc_dict=enc_dict,
        metadata=model_info,
        test_data=test_data
    )

    ## Save to Redshift
    logging.info('Saving model info to Redshift...')
    run_from_template(
        template_configs={
            'template': configs['sql']['model_info'],
            'snapshot_date': configs['model']['snapshot_date'],
            'id': model_id,
            **model_info
        },
        redshift_configs=configs['sql']['redshift']
    )

    ## FIT FINAL MODEL ON ALL DATA
    ##############################
    if final_model:
        logging.info('Fitting final model on entire data...')
        final_model_dict = {m: model_dict[m] for m in [results['model']]}
        try:
            final_model, results, model_cols = fit_model(
                x=dat.drop(labels=target, axis=1),
                y=dat[target],
                model=final_model_dict,
                data_configs=configs['data'],
                preprocessing_configs=configs['model']['preprocessing'],
                time_sampling_configs=configs['model']['time_sampling'],
                model_configs=configs['model'],
                training_configs=configs['training'],
                model_type='final'
            )
        except Exception as e:
            logging.exception('Failure fitting final model. Skipping...')
            raise e

        ## Update configs
        configs['model']['model_cols'] = model_cols

        ## SAVE FINAL MODEL
        ##############################
        logging.info('Saving final model to disk...')
        model_info = mk_info(
            type='final',
            algo=results['model'],
            version=0.1,
            target=target
        )

        save_model_output(
            model=final_model,
            configs=configs,
            enc_dict=enc_dict,
            metadata=model_info)

def train_models(configs):
    """
    Load dataset from Redshift, extract targets, and train model
    for each of the targets. For targets that fail being fit, then
    save target to file

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

    ## Make model dictionary from training configs
    model_dict = mk_model_dict(configs['training'])

    ## TRUNCATE PROCESS FILES
    ##############################
    ## If a target fails to score, we skip it
    ## and save the target to a file
    ## At the beginning of running this
    ## script, we clear the file
    if os.path.exists(configs['output']['failed_targets']['build']):
        os.remove(configs['output']['failed_targets']['build'])

    ## LOAD SAMPLE DATA
    ##############################
    logging.info('Extracting model data from Redshift...')
    dat = extract_data(
        configs['sql']['redshift'],
        configs['sql']['tables']['modeling']
    )
    logging.info('Successfully extracted data of size {}...'.format(dat.shape))

    ## IMPUTATION
    ##############################
    # The only imputation this script does
    # is to check for missing values, and upon
    # finding any raise a warning and fill with 0.
    logging.info('Checking for and imputing missing values...')
    dat = check_missing_vals(dat)

    ## COLUMN ENCODING
    ##############################
    logging.info('Encoding categorical columns...')
    dat_enc, enc_dict = encode_cols(
        dat, do_ohe=configs['model']['preprocessing']['do_onehotencoding'])

    ## UPDATE CONFIGS
    ##############################
    ## Add categorical columns to configs
    ## to be saved with the results
    if not configs['model']['preprocessing']['do_onehotencoding']:
        configs['data']['cat_cols'] = list(enc_dict.keys())

    ## Add snapshot date
    configs['model']['snapshot_date'] = max(dat[configs['data']['date']])

    ## Add algo-specific configs
    if 'LightGBM' in model_dict:
        configs['training']['LightGBM']['fit_params'].update(
            {'categorical_feature': list(enc_dict.keys())}
        )

    ## REMOVE INCOMPLETE WEEKS
    ##############################
    ## The most recent weeks have targets
    ## with incomplete information.
    ## For example we have data on the current
    ## week, but we don't know we happen
    ## next week, which is in the future.
    ## Including that data in the
    ## training set will induce noise
    logging.info('Removing recent weeks without complete target information')
    dat_fit = rm_incomplete_weeks(
        dat_enc,
        configs['data']['weeknum'],
        configs['model']['time_sampling']['pred_horizon'],
        configs['model']['time_sampling']['pred_gap']
    )

    ## FIT MODELS FOR EACH TARGET
    ##############################
    targets = extract_targets(dat, ignore=configs['data']['global_target'])
    configs['data']['targets'] = targets

    for n, target in enumerate(targets):

        logging.info('Using target {} of {}: {}'.format(n, len(targets), target))
        try:
            train_target(dat_fit, target, model_dict, enc_dict, configs)
        except Exception as e:
            logging.exception('Failure fitting target {}'.format(target))
            ## If target fails in any way then we save it to a file
            with open(configs['output']['failed_targets']['build'], 'a') as f:
                f.write(target + '\n')
            continue

# --------------------------------------------------------------------
# Default Configurations
# --------------------------------------------------------------------

CONFIGS = {
    'working_dir': '/path/to/project/root',
    'config_files': {
        'sql': 'configs/configs_sql.json',
        'data': 'configs/configs_data.json',
        'model': 'configs/configs_model.json',
        'training': 'configs/configs_training.json',
        'output': 'configs/configs_output.json'
    }
}

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == '__main__':

    train_models(configs=CONFIGS)
