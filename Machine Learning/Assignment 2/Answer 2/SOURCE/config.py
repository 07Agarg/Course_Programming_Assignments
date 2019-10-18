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

INPUT_FILE_PATH = "breast-cancer-wisconsin.data"
LABELS_FILE_PATH = "breast-cancer-wisconsin.names"

FILE2 = "wdbc.data"
FILE3 = "wdbc.names"

LEARNING_RATE = 0.001

NUM_EPOCHS = 30

#CROSS VALIDATION PARAMETER
K_FOLDS = 5

SEED = 42
