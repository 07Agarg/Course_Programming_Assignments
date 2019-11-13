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
NUM_TRAIN = 50000
NUM_TEST = 10000
IMAGE_SIZE = 28
CLASSES = 10
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

TRAIN_INPUT = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"

TEST_INPUT = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

#Model Parameters
LEARNING_RATE = 0.1
NUM_EPOCHS = 10000

SEED = 128
