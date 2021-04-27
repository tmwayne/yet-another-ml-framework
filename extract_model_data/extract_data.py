#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
escription: connect to redshift to extract data
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

import sys
import json
import argparse
import psycopg2
import pandas.io.sql as sqlio

import logging
# logging = logging.getLogger("buildLogger")

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def get_data(redshift_configs, query):
    """ Run a SQL query and collect the output in a pandas DataFrame """
    assert all(x in redshift_configs for x in ['host', 'db', 'port', 'user'])

    try:
        logging.info('  Connecting to Redshift...')
        conn = psycopg2.connect(
            host=redshift_configs['host'],
            dbname=redshift_configs['db'],
            port=redshift_configs['port'],
            user=redshift_configs['user']
        )
    except Exception as e:
        logging.exception('Failure connecting to Redshift')
        raise e

    try:
        logging.info('  Extracting data...')
        dat = sqlio.read_sql_query(query, conn)
    except Exception as e:
        logging.exception('Failure extracting data from Redshift')
        raise e
    finally:
        conn.close()

    return dat

def extract_data(redshift_configs, table):

    query = 'select * from {table}'.format(table=table)
    return get_data(redshift_configs, query)

def extract_names(redshift_configs, table):

    query = 'select * from {table} limit 0'.format(table=table)
    column_names = get_data(redshift_configs, query)
    return list(column_names.columns)

# --------------------------------------------------------------------
# Default Configurations
# --------------------------------------------------------------------

configs = {
    'working_dir': '/path/to/project/root',
    'sql_configs': 'configs/configs_sql.json',
    'sql': {
        'host': '',
        'db': 'analytics',
        'port': 5439,
        'user': 'user',
        'table': 'table'
    }
}

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract modeling data from Redshift')
    parser.add_argument('--sql-configs', nargs=1, default=[False], help='SQL configurations')
    parser.add_argument('--table', nargs=1, default=[False], help='Table to extract from Redshift')

    args = parser.parse_args()

    ## include project wide configurations
    ## this is where the redshift connection information is stored
    if args.sql_configs[0]:
        configs['sql_configs'] = args.sql_configs[0]

    with open(configs['sql_configs'], 'r') as f:
        sql_configs = json.load(f)
        configs = {**configs, **sql_configs}

    if args.sql_configs[0]:
        configs['table'] = args.table[0]

    dat = extract_data(configs['sql'], configs['sql']['table'])
    dat.to_csv(sys.stdout, index=False)

