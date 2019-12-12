# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 03:36:55 2019

@author: Ashima
"""


import os

#Directory Information
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, "DATASET/")
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')

TRAIN_FILE = "train_CIFAR.pickle"
TEST_FILE = "test_CIFAR.pickle"

TRAIN_FEATURES = "train_features.pickle"
TEST_FEATURES = "test_features.pickle"

#Dataset Information
IMAGE_SIZE = 32
CLASSES = 10
CHANNELS = 3

NUM_SAMPLES = 10000
NUM_TEST = 2000 
#SPLIT_FACTOR = 25
#NUM_VAL = (SPLIT_FACTOR/100.)*NUM_SAMPLES
#NUM_TRAIN = int(NUM_SAMPLES - NUM_VAL)

SEED = 128

CLASS_LABELS = [0, 1]