# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:49:20 2019

@author: Ashima
"""


import os

#Directory Information
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, "DATASET/")
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')

#Dataset Information
NUM_SAMPLES = 50000
NUM_TEST = 10000
IMAGE_SIZE = 28
CLASSES = 10
CLASS_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SPLIT_FACTOR = 25
NUM_VAL = (SPLIT_FACTOR/100.)*NUM_SAMPLES
NUM_TRAIN = (NUM_SAMPLES - NUM_VAL)

TRAIN_INPUT = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"

TEST_INPUT = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

#Model Parameters
LEARNING_RATE = 0.9
NUM_EPOCHS = 50

SEED = 128
