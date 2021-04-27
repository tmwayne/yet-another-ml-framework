#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
description: fit a model, with optional model selection and tuning
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score

import logging
# logging = logging.getLogger("buildLogger")

from modules.src.build_model_pipeline import outOfTimeTuner


# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def get_row_indices(df, col, vals):
    """ return the row indices for columns values that match a list """

    return list(df[df[col].isin(vals)].index)

def get_cv_indices(df, col, time_slices):
    """ get the row indices of the train and holdout set
    for each time slice """
    return [
        ## get train and holdout indices for slice
        tuple(get_row_indices(df, col, slc[x]) for x in range(2))

        ## get indices for each slice
        for slc in time_slices
    ]

def cv_sizes(cv_indices):
    return [(len(x[0]), len(x[1])) for x in cv_indices]

def create_time_slices(weeks, lookback, horizon, gap,
    step_size, holdout_window, num_steps):
    """
    calculate number of time slices given the length of training data,
    size of lookback window, prediction horizon, and step size

    :param n: length of date range in training data
    :type int

    :param lookback: maximum size of lookback window
    :type int

    :param horizon: size of prediction horizon
    :type int

    :param step_size: size of desired step size for time slices
    :type int
    ...
    raises exception: length of training data isn't long enough
    ...
    :return: number of time slices possible
    :rtype: int
    """

    n = len(weeks)
    min_week = min(weeks)
    holdout_gap = horizon + gap - 1 # gap between train and holdout set
    holdout_size = horizon + holdout_window - 1
    step_space = (num_steps - 1) * step_size

    training_window = n - lookback - holdout_gap - holdout_size - step_space

    if training_window <= 0:
        err_msg = "negative window size using specified parameters"
        logging.error(err_msg)
        raise Exception(err_msg)

    def create_time_slice(step=0):
        base = min_week + lookback + step
        time_slice = (
            [base + x for x in range(training_window)],
            [base + x + holdout_gap + training_window
            for x in range(holdout_window)]
        )
        return time_slice

    output = [create_time_slice(x*step_size) for x in range(0, num_steps)]

    return output

def select_best_results(results, metric='mean_auc', best="max"):
    """ select the model with the best metric from a results dictionary """
    best_metric = 0
    best_model = ''
    for model, result in results.items():
        if best == "max":
            is_better = result[metric] > best_metric
        elif best == "min":
            is_better = result[metric] < best_metric
        else:
            raise Exception('best must be either min or max')

        if is_better:
            best_metric = result[metric]
            best_model = model

        output = {
            'model': best_model,
            'hyper_params': results[best_model]['hyper_params'],
            metric: best_metric
        }

    return output

def fit_model(x, y, model, data_configs, preprocessing_configs,
    time_sampling_configs, model_configs, training_configs,
    model_type='tuning'):
    """
    Downsamples dataset, does feature selection using Lasso,
    then either does out-of-time hyper-parameter tuning and
    model selection for fits a single model with specified
    hyperparameters

    :param x: modeling dataset, must already have missing values
        imputed and categorical columns one-hot encoded
    :type DataFrame

    :param y: target variable of same length as x
    :type Series

    :param model: modeling class if model_type is final, otherwise
        a dictionary that maps the model name in the tuning-grid
        to the modeling class it refers to. Must have both
        a fit method and a predict method. Hyper-parameters
        will be passed via the tuning-grid
    :type model class or model dictionary

    :param preprocessing_configs: configurations for down-sampling
    :type dict with element downsample_ratio

    :param time_sampling_configs: configurations for setting up
        out-of-time sampling
    :type dict with elements holdout_window, max_lookback, num_steps,
        pred_horizon, step_size

    :param model_configs: configurations for modeling including tuning grid
    :type dict with elements random_seed, tuning_grid

    :param model_type: setting for doing hyper-parameter tuning and
        model selection or just fitting a model with given hyper-params
    :type str of either tuning or final

    :return: final model fit to entire data set after downsampling
    :rtype: class of best model type

    :return: results of the best model
    :rtype: dict with elements model, hyper_parameters, mean_auc

    :return: columns used for training model
    :return list
    """

    err_msg = 'Error: model_type must be either tuning or final'
    assert any(x in model_type for x in ['tuning', 'final']), err_msg

    x = x.copy()

    ## REMOVE NON-MODELING COLS
    ##############################
    logging.info('  Removing non-modeling cols...')
    try:
        x.drop(labels=data_configs['remove_cols'], axis=1, inplace=True)
    except Exception as e:
        logging.exception('Failure dropping non-modeling cols')
        raise e

    ## CLASS IMBALANCE
    ##############################
    ## We test to see if the dataset even needs to be downsampled
    cur_ratio = sum(y) / sum(y == 0)
    ds_ratio = preprocessing_configs['downsample_ratio']

    if preprocessing_configs['do_downsampling'] and cur_ratio < ds_ratio:
        try:
            logging.info('  Downsampling to balance classes...')
            msg = '  Shape of data {} downsampling: {}'
            logging.info(msg.format('before', x.shape))

            rus = RandomUnderSampler(
                sampling_strategy=ds_ratio,
                random_state=model_configs['random_seed'])
            x_res, y_res = rus.fit_sample(x, y)

            ## Convert x_res back to DataFrame
            ## so that we can filter it using column names
            x_res = pd.DataFrame(x_res, columns=x.columns)
            logging.info(msg.format('after', x_res.shape))

        except Exception as e:
            logging.exception('Failure downsampling training data')
            raise e

    else:
        x_res, y_res = x, y

    ## FEATURE SELECTION
    ##############################
    if preprocessing_configs['do_feature_selection']:
        logging.info('  Selecting features with Lasso...')
        try:
            clf = Lasso(alpha=0.0005)
            clf.fit(x_res, y_res)

            model_cols = list(x.columns[clf.coef_ > 0])
            logging.info('\tSelected {} columns'.format(len(model_cols)))

            ## If LightGBM is one of the models, then we need to update
            ## the list of categorical features
            if 'LightGBM' in model.keys():
                lightgbm_fit_params = training_configs['LightGBM']['fit_params']
                lightgbm_fit_params['categorical_feature'] = [
                    col for col in lightgbm_fit_params['categorical_feature']
                    if col in model_cols
                ]

        except Exception as e:
            logging.exception('Failure executing Lasso feature selection')
            raise e
    else:
        logging.info('  Skipping feature selection...')
        model_cols = x.columns

    ## TUNING MODEL
    ##############################
    if model_type == 'tuning':

        err_msg = "If model_type='tuning' then model must be a dictionary"
        assert isinstance(model, dict), err_msg

        ## OUT-OF-TIME SAMPLING
        ##############################
        ## If the model is for tuning hyperparameters
        ## then we do cross-validation using out-of-time
        ## sample. Otherwise we fit using either
        ## the provided or default hyperparameters
        logging.info('  Creating out-of-time sampling indices...')
        try:
            week_nums = x[data_configs['weeknum']].unique()

            ## Get the week numbers for each of the time slices
            time_slices = create_time_slices(
                weeks=week_nums,
                lookback=time_sampling_configs['max_lookback'],
                horizon=time_sampling_configs['pred_horizon'],
                gap=time_sampling_configs['pred_gap'],
                step_size=time_sampling_configs['step_size'],
                holdout_window=time_sampling_configs['holdout_window'],
                num_steps=time_sampling_configs['num_steps']
            )

            ## Get the row indices for each of the time slices
            cv_indices = get_cv_indices(
                x_res, data_configs['weeknum'], time_slices)
        except Exception as e:
            logging.exception('Failure creating time slices')
            raise e

        ## CROSS-VALIDATION
        ##############################
        results = {}
        logging.info('  Tuning hyper-parameters using out-of-time sampling...')
        for model_name in model.keys():
            try:
                train_params = training_configs[model_name]
                tuning_model = model[model_name]
            except KeyError as e:
                err_msg = 'Model {} not found in training configs'.format(model_name)
                logging.exception(err_msg)

            try:
                logging.info('  Tuning model {}...'.format(model_name))
                oots = outOfTimeTuner(tuning_model, train_params, 'auc')
                oots.tune_model(x_res[model_cols], y_res, cv_indices)
                # print(oots.get_results())
                best_results = oots.get_best_hyperparams()

                results[model_name] = best_results

            except Exception as e:
                logging.exception('Failure tuning hyper-parameters')
                raise e

        ## Here we select the best model
        best_results = select_best_results(results)
        logging.info('  Best model is {} using {} and has {} of {}'.format(
            best_results['model'],
            best_results['hyper_params'],
            'mean_auc',
            best_results['mean_auc']))

        ## FIT FINAL TUNING MODEL
        ##############################
        logging.info('  Fitting final tuning model...')
        try:
           final_model = model[best_results['model']](**best_results['hyper_params'])
           final_model.fit(x[model_cols], y)
        except Exception as e:
            logging.exception('Failure fitting final model')
            raise e


    ## SINGLE MODEL
    ##############################
    elif model_type == 'final':
        try:
            tuning_results = model_configs['tuning_results']
            model_name = tuning_results['model']
            training_params = training_configs[model_name]
            model = model[model_name]

            final_model = model(
                **training_params['model_params'],
                **tuning_results['hyper_params'])
            final_model.fit(x[model_cols], y, **training_params['fit_params'])

            best_results = tuning_results.copy()
            best_results['tuning_auc'] = best_results.pop('mean_auc')

        except Exception as e:
            logging.exception('Failure fitting final model')
            raise e

    return final_model, best_results, list(model_cols),


