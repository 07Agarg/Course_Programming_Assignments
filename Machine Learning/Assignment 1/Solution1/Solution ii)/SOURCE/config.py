# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:54:57 2019

@author: ashima
"""

import os

DATASET = "DATASET/"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, DATASET)
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')

REGULARIZATION_PARAM = "REGULARIZATION_PARAM.txt"

LEARNING_RATE = 0.001
LAMBDA_L2 = 0.000001
LAMBDA_L1 = 0.00001

TRAIN_PATH = "Train.csv"
TEST_PATH = "Test.csv"

NUM_EPOCHS = 30

#CROSS VALIDATION PARAMETER
K_FOLDS = 5

SEED = 128
