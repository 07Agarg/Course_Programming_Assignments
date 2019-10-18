# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:57:09 2019

@author: ashima
"""

import os

#DIRECTORY INFORMATION
DATASET = "DATASET/cifar-10-python/"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, DATASET)
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')

TRAIN_FILE1 = "data_batch_1"
TRAIN_FILE2 = "data_batch_2"
TRAIN_FILE3 = "data_batch_3"
TRAIN_FILE4 = "data_batch_4"
TRAIN_FILE5 = "data_batch_5"

LABEL_NAMES = "batches.meta"

TEST_FILE = "test_batch"


RMSE_NORMAL_FILE = "RMSE_LOSS_NORMAL.txt"             #RMSE on Training Set Using Normal Equation
RMSE_GRAD_FILE = "RMSE_LOSS_GRAD.txt"                 #RMSE on Training Set Using Gradient Descent
RMSE_VAL_NORMAL_FILE = "RMSE_VAL_NORMAL_FILE.txt"     #RMSE on Validation Set Using Normal Equation
RMSE_VAL_GRAD_FILE  = "RMSE_VAL_GRAD_FILE.txt"        #RMSE on Validation Set Using Gradient Descent


LEARNING_RATE = 0.001

NUM_EPOCHS = 30

#CROSS VALIDATION PARAMETER
K_FOLDS = 5

SEED = 128
