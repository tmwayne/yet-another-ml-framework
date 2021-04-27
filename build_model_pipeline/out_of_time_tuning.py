#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
description: hyper-parameter tuning using out-of-time sampling
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

from collections import OrderedDict
from itertools import product

import pandas as pd
from numpy import mean
from sklearn.metrics import roc_auc_score

import logging
# logging = logging.getLogger("buildLogger")

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def create_tuning_grid(hyper_params):
    """ Create a tuning grid from a dictionary
    that specifies possible values """
    ## We don't want to force users to enter
    ## single tuning param values as lists.
    ## Therefore if the value isn't a list,
    ## we convert it to one
    for k,v in hyper_params.items():
        hyper_params[k] = v if isinstance(v, list) else [v]

    ## We want to avoid any weird bugs owing
    ## from a dictionary guarantee order.
    ## Therefore we use OrderedDict to guarantee order
    hyper_params = OrderedDict(hyper_params)
    tuning_grid = [
        {k:v for k,v in zip(hyper_params.keys(), hp_set)}
        for hp_set in product(*hyper_params.values())
    ]
    return tuning_grid

class outOfTimeTuner():
    """
    Perform hyper-parameter tuning given a model,
    a set of hyper-parameters, and a set of out-of-time indices
    """

    def __init__(self, model, train_params, metric):
        self.model = model
        self.model_params = train_params['model_params']
        self.fit_params = train_params['fit_params']
        self.metric = metric
        self.tuning_grid = create_tuning_grid(train_params['hyper_params'])
        self.tuning_results = {
            'hp_ind' + str(n): {'hyper_params': hp, 'results': []}
            for n, hp in enumerate(self.tuning_grid)
        }

    def _split_data(self, x, y, ind_set):
        self._x_train = x.loc[ind_set[0],]
        self._y_train = y[ind_set[0]]
        self._x_test = x.loc[ind_set[1],]
        self._y_test = y[ind_set[1]]

    def _fit_slice(self, hyper_params):
        """ Pass the hyper params to model instantiation
            and model fitting """
        self._clf = self.model(**self.model_params, **hyper_params)
        self._clf.fit(self._x_train, self._y_train, **self.fit_params)

    def _eval_test(self, hp_ind):
        pred_test = self._clf.predict_proba(self._x_test)[:,1]
        auc_test = round(roc_auc_score(self._y_test, pred_test), 4)
        self.tuning_results[hp_ind]['results'].append(auc_test)
        return auc_test

    def tune_model(self, x, y, indices):
        """ Fit a model for each slice in the collection
        of indices and for each set of hyperparameters """
        try:
            for n, ind_set in enumerate(indices):
                self._split_data(x, y, ind_set)

                for hp_ind, hp in self.tuning_results.items():
                    self._fit_slice(hp['hyper_params'])
                    auc_test = self._eval_test(hp_ind)
                    logging.info('  HPs: {}  Slice: {}  AUC: {}'.format(
                        hp['hyper_params'], n, auc_test))
        except Exception as e:
            err_msg = 'Failure fitting model with Hyper-param {} on slice {}'.format(hp_ind, n)
            logging.exception(err_msg)
            raise e

    def get_results(self):
        results = {}
        for ind, result in self.tuning_results.items():
            results[ind] = {
                'hyper_params': result['hyper_params'],
                'mean_' + self.metric: round(
                    mean(result['results']), 4)
            }
        return results

    def get_best_hyperparams(self):
        best_metric = 0
        best_ind = ''
        results = self.get_results()
        for ind, result in results.items():
            metric = result['mean_' + self.metric]
            if metric > best_metric:
                best_metric = metric
                best_ind = ind

        return results[best_ind]

    def __repr__():
        pass

