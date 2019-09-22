# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:54:57 2019

@author: ashima
"""

import os
#Directory Information
DATASET = "DATASET/"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, DATASET)
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

NOREG_WEIGHTS = "NoReg_Weights.txt"
L1_WEIGHTS = "L1_Weights.txt"
L2_WEIGHTS = "L2_Weights.txt"

#HyperParameters
NUM_EPOCHS = 200
LEARNING_RATE = 1
LAMBDA = 1
#LAMBDA_L1 = 1
#LAMBDA_L2 = 1


SEED = 128
