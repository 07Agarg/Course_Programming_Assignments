# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 05:02:24 2019

@author: Ashima
"""

import os

#DIRECTORY INFORMATION
DATASET = "DATASET/"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, DATASET)
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')

INPUT_FILE_PATH = "wdbc.data"

SEED = 42

DECISION_TREE_DEPTH = 10
