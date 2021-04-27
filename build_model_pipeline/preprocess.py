#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
description: functions for preprocessing data
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

from collections import defaultdict

import pandas as pd
from numpy import setdiff1d, append
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from modules.src.helper_functions import timeit

import logging
# logging = logging.getLogger("buildLogger")

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def check_missing_vals(dat):
    """
    Check for missing values in DataFrame.
    If any are found log a warning, then fill with zero.

    :param dat: training data
    :type pd.core.frame.DataFrame
    :return: DataFrame with null values imputed to zero
    :rtype: pd.core.frame.DataFrame
    """

    df = dat.copy()
    num_nulls = df.isna().sum()
    missing_vals = list(num_nulls.index[num_nulls > 0])

    if len(missing_vals) > 0:
        logging.warn('  Missing values identified in training data')
        logging.warn(num_nulls[num_nulls > 0])

        logging.info('  Imputing missing values to be 0 or MISSING...')

        ## Categorical columns are filled with MISSING
        ## Filling with 0 will cause issues with LabelEncoder
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].fillna('MISSING')

        num_cols = df.select_dtypes(include=['int', 'float']).columns
        df[num_cols] = df[num_cols].fillna(0)

        remaining = list(df.dtypes[df.isna().sum() > 0].unique())
        if len(remaining) > 0:
            err_msg = 'Failuring imputing values for types {}'.format(remaining)
            log.error(err_msg)
            raise Exception(err_msg)

    return df

def encode_cols(df, enc_dict=None, do_ohe=False):
    """
    One-hot encoding for training or scoring data.
    If used for scoring, a previously fit label encoding
    dictionary should be provided

    :param df: training or scoring dataframe
    :type pd.core.frame.DataFrame
    :param enc_dict: dictionary of LabelEncoders (optional)
    :type collections.defaultdict with values LabelEncoders
    ...
    raises Exception: if enc_dict is provided and at least one
        column is missing from df
    ...
    :return: dataframe with one-hot encoded columns
    :rtype: pd.core.frame.DataFrame
    :return: dictionary of label encoders to be used for scoring
    :rtype: collections.defaultdict with values LabelEncoders
    """
    dat = df.copy().reset_index(drop=True)

    cat_cols = dat.select_dtypes(include=['category', 'object']).columns

    ## This dictionary will store the label encodings
    ## for each of the columns. This will allow us to
    ## use the same encodings for new data that we score
    if enc_dict:
        ## If the user provides an enc_dict then we only
        ## one-hot encode columns that had been encoded
        ## previously
        enc_provided = True

        ## If any of the categorical columns from the enc_dict
        ## are missing in the data, then raise an error because
        ## the model needs all columns to be able to fit
        missing_cols = [c for c in enc_dict.keys() if c not in cat_cols]
        if len(missing_cols) > 0:
            err_msg = 'Missing categorical columns {} in scoring data'.format(
                ', '.join(missing_cols))
            logging.error(err_msg)
            raise Exception(err_msg)

        cat_cols = [c for c in cat_cols if c in enc_dict.keys()]

    else:
        enc_provided = False
        enc_dict = defaultdict(LabelEncoder)

    def new_cols(col, cats, sep='_'):
        return [sep.join([col, cat]) for cat in cats]

    if len(cat_cols) > 0:

        ohe_cols = []
        for col in cat_cols:
            if not enc_provided:
                enc_dict[col].fit(dat[col])
            else:
                ## If an encoder has been provided
                ## then we must add any new values to
                ## the encoder to prevent it from
                ## throwing an error about unseen values
                _diff = setdiff1d(dat[col].unique(), enc_dict[col].classes_)
                if _diff.size > 0:
                    enc_dict[col].classes_ = append(enc_dict[col].classes_, _diff)

            dat[col] = enc_dict[col].transform(dat[col])
            ## These will be the names of the new columns we create
            # ohe_cols += new_cols(col, enc_dict[col].classes_)

        if do_ohe:
            logging.info('  One-hot encoding columns...')
            ohe = OneHotEncoder(categories='auto')
            dat_ohe = ohe.fit_transform(dat[cat_cols]).toarray()
            df_encoded = pd.DataFrame(dat_ohe, columns=ohe_cols, dtype=int)
        else:
            df_encoded = dat[cat_cols]

        ## The encoded columns are appended
        ## to the non-categorical columns to be returned
        output = pd.concat(
            [df.reset_index(drop=True).select_dtypes(
                exclude=['object', 'category']),
            df_encoded],
            axis=1
        )

    else:
        logging.info('  No category columns, returning original dataset')
        output = dat

    return output, enc_dict
