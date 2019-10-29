# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:57:09 2019

@author: Ashima
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

#CROSS VALIDATION PARAMETER
K_FOLDS = 5

SEED = 128

TRAIN_SIZE = 50000
TEST_SIZE = 10000
CHANNELS = 3
IMG_SIZE = 32
HOG_FEATURES = 144           ##Convert 32*32*3 (1024) to 324 feature vector

SHRINK_SIZE = 16

SUPPORT_DATA = "support_data"
SUPPORT_LABELS = "support_labels"