# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:54:57 2019

@author: ashima
"""

import os

DATASET = "DATASET/Dataset.data/"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, DATASET)
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')

DATASET_LOW_RMSE = '../Solution ii)/DATASET/'
DATASET_OUTDIR = os.path.join(ROOT_DIR, DATASET_LOW_RMSE)

RMSE_NORMAL_FILE = "RMSE_LOSS_NORMAL.txt"             #RMSE on Training Set Using Normal Equation
RMSE_GRAD_FILE = "RMSE_LOSS_GRAD.txt"                 #RMSE on Training Set Using Gradient Descent
RMSE_VAL_NORMAL_FILE = "RMSE_VAL_NORMAL_FILE.txt"     #RMSE on Validation Set Using Normal Equation
RMSE_VAL_GRAD_FILE  = "RMSE_VAL_GRAD_FILE.txt"        #RMSE on Validation Set Using Gradient Descent

LEARNING_RATE = 0.001

FILE_PATH = "Dataset.data"
#TEST_PATH = "test_iris.csv"

NUM_EPOCHS = 30

#CROSS VALIDATION PARAMETER
K_FOLDS = 5

SEED = 128
