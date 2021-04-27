#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
description: run a query by passed filling in a template
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

import os
import configparser
import json
import psycopg2
import boto3
from time import time

import logging
# logging = logging.getLogger("scoreLogger")

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def mk_tmp_file(dir='/tmp'):
    """ Return the name of a temp file """
    return os.path.join(dir, str(round(time())))

def from_home(path='~'):
    return path.replace('~', os.path.expanduser('~'))

def read_aws_creds(path=None):
    """ Load aws configs from file """
    path = path or os.path.join(from_home(), '.aws/credentials')
    aws_creds = configparser.ConfigParser()

    try:
        aws_creds.read(path)
        creds = aws_creds['default']
    except:
        raise Exception('Default aws credentials not found')

    return creds

def read_json(file):
    try:
        with open(file, 'r') as f:
            configs = json.load(f)
    except:
        raise Exception('Unable to load json file')

    return configs

def read_query(path):
    try:
        with open(path, 'r') as f:
            query = f.read()
    except:
        raise Exception('Failure loading query')

    return query

def fill_template(template, configs):
    try:
        return template.format(**configs)
    except:
        raise Exception('Unable to fill template')

def run_query(redshift_configs, query):
    """ Connect to Redshift and execute a single query
    and ignore any output """
    assert all(x in redshift_configs for x in ['host', 'db', 'port', 'user'])
    ## connect to redshift and execute query
    try:
        logging.info('  Creating connection to redshift...')
        conn = psycopg2.connect(
            host=redshift_configs['host'],
            dbname=redshift_configs['db'],
            port=redshift_configs['port'],
            user=redshift_configs['user']
        )

        cur = conn.cursor()
        logging.info('  Executing query...')
        cur.execute(query)
        conn.commit()
        cur.close()
    except Exception as e:
        logging.exception('Failure executing query on redshift')
        raise e
    finally:
        conn.close()

    logging.info('  Successfully ran query!')

def run_copy_query(template_configs, s3_configs,
    redshift_configs, aws_creds_path=None):

    ## Load template
    template = read_query(template_configs['template'])

    ## Read in aws credentials to fill in template
    aws_creds = read_aws_creds(aws_creds_path)

    ## Combine all template configurations
    template_configs = {
        **s3_configs,
        **template_configs,
        **aws_creds
    }

    ## Complete the template with configurations
    query = fill_template(template, template_configs)

    ### Run query
    run_query(redshift_configs, query)

def save_to_s3(dat, path, bucket, **kwargs):
    """ Save output data to S3 bucket """
    tmp_file = mk_tmp_file()

    ## Save to tmp file
    try:
        dat.to_csv(
            path_or_buf=tmp_file,
            sep='|',
            header=False,
            index=False,
            compression='gzip',
            **kwargs
        )
    except Exception as e:
        logging.exception('Failuring saving output to tmp file')
        raise e

    ## Copy to S3
    try:
        s3_resource = boto3.resource('s3')
        s3_object = s3_resource.Object(bucket.replace('s3://', ''), path)
        s3_object.upload_file(tmp_file)
    except Exception as e:
        logging.exception('Failure copying data to S3 bucket')
        raise e
    finally:
        os.remove(tmp_file)

def run_from_template(template_configs, redshift_configs):
    """ Run a query by first loading and filling out a template """
    template = read_query(template_configs.pop('template'))
    query = fill_template(template, template_configs)
    run_query(redshift_configs, query)

