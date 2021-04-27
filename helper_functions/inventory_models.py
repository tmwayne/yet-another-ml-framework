#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

"""
description: print information on output packages
author: Tyler Wayne
last modified: 2021-04-27
"""

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

import os
import json
import argparse
from time import gmtime, strftime
from tabulate import tabulate

# --------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------

def inventory_models(output_dir, type=['tuning', 'final'], pprint=False):
    """ Reads the output packages in the output directory
    and prints inventory to console """

    oldcwd = os.getcwd()
    os.chdir(output_dir)

    pkg_info = {}
    for pkg_id in os.listdir():
        try:
            with open(pkg_id + '/info.json', 'r') as f:
                info = json.load(f)
            pkg_info.update({pkg_id: info})
        except Exception as e:
            continue

    if pprint:
        output = [['id', 'date', 'target', 'type', 'algo']]
        for pkg_id, info in pkg_info.items():
            try:
                if info['type'] in type:
                    info_row = [
                        pkg_id,
                        strftime('%Y-%m-%d', gmtime(info['date_created'])),
                        info['target'],
                        info['type'],
                        info['algo'],
                    ]
                    output.append(info_row)

            except Exception as e:
                raise Exception('Corrupted model package: {}'.format(pkg_id))
    else:
        output = pkg_info

    os.chdir(oldcwd)
    return output

# --------------------------------------------------------------------
# Default Configurations
# --------------------------------------------------------------------

CONFIGS = {}

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == '__main__':

    ## Set up arg parser
    parser = argparse.ArgumentParser(
        description='Take inventory of modeling output packages')
    parser.add_argument('--dir', nargs=1,
        default=['output/model'], help='Output directory')
    parser.add_argument(
        '--type', nargs=1, choices=['tuning', 'final'],
        default=[['tuning', 'final']], help='Model type')
    parser.add_argument(
        '--sort', nargs=1, choices=['date', 'algo', 'auc'],
        help='Sort inventory by specified column')

    ## Parse args
    args = parser.parse_args()

    CONFIGS['output_dir'] = args.dir[0]
    CONFIGS['model_type'] = args.type[0]

    inventory = inventory_models(
        CONFIGS['output_dir'],
        type=CONFIGS['model_type'],
        pprint=True
    )

    print(tabulate(inventory, headers='firstrow'))

