# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:57:09 2019

@author: Ashima
"""
import time
import cv2
import pickle
import pandas as pd
import config
import os
import numpy as np
from skimage.feature import hog

class Data:
    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.size = None 
        self.dataX_hog = None
#    def preprocess(self):
#        maxs = np.max(self.dataX)
#        mins = np.min(self.dataX)
#        self.dataX = (self.dataX - mins)/(maxs - mins)
    def extract_hog_features(self, img):
        #print("img shape: ", img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_hog = hog(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), block_norm='L2-Hys', visualize=False, transform_sqrt=True, feature_vector=True, multichannel=None)
        #print("hog feature shape: ", img_hog.shape)
        #print(img_hog.shape)
        return img_hog
    
    def preprocess_data_hog(self, data_size):
        print("dataX shape: ", self.dataX.shape)
        print("dataY shape: ", self.dataY.shape)
        self.dataX = np.reshape(self.dataX, (data_size, config.CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
        print("dataX reshaped shape: ", self.dataX.shape)
        self.dataX = np.array(self.dataX)
        self.dataX = self.dataX.transpose(0, 2, 3, 1)
        self.dataX_hog = np.zeros((data_size, config.HOG_FEATURES))
        start_time = time.time()
        for i in range(data_size):
            self.dataX_hog[i] = self.extract_hog_features(self.dataX[i])
        end_time = time.time()
        print("feature extraction time: {} sec".format(end_time - start_time))
        print(self.dataX_hog.shape)
#        print("dataX reshaped shape now : ", self.dataX.shape)
#        cv2.imshow('first image', self.dataX[1])
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
    def preprocess_data_shrink(self, data_size):
        print("dataX shape: ", self.dataX.shape)
        print("dataY shape: ", self.dataY.shape)
        self.dataX = np.reshape(self.dataX, (data_size, config.CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
        print("dataX reshaped shape: ", self.dataX.shape)
        self.dataX = np.array(self.dataX)
        start_time = time.time()
        self.dataX = self.dataX.transpose(0, 2, 3, 1)
        for i in range(data_size):
            self.dataX[i] = cv2.resize(cv2.cvtColor(cv2.cvtColor(self.dataX[i], cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY), (config.SHRINK_SIZE, config.SHRINK_SIZE))
        end_time = time.time()
        print("feature extraction time: {} sec".format(end_time - start_time))
        print(self.dataX.shape)
    
    def read(self, filename):
        file = os.path.join(config.DATA_DIR, filename)
        with open(file, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            if self.dataX is None:
                self.dataX = dict[b'data']
                self.dataY = dict[b'labels']
                self.dataX = np.asarray(self.dataX)
                self.dataY = np.asarray(self.dataY)
                #self.dataY = self.dataY.reshape(self.dataY.shape[0], 1)
            else:
                dataX = dict[b'data']
                dataY = dict[b'labels']
                dataX = np.asarray(dataX)
                dataY = np.asarray(dataY)
                #dataY = dataY.reshape(dataY.shape[0], 1)
                self.dataX = np.vstack((self.dataX, dataX))
                self.dataY = np.append(self.dataY, dataY)
                print(self.dataY.shape)
                
    def read_test(self, filename):
        file = os.path.join(config.DATA_DIR, filename)
        with open(file, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            self.dataX = dict[b'data']
            self.dataY = dict[b'labels']
            self.dataX = np.asarray(self.dataX)
            self.dataY = np.asarray(self.dataY)

    def get_data(self):
        #self.preprocess()
        return self.dataX_hog, self.dataY
    
    def save_data(self):
        train = pd.DataFrame(self.train_set)
        test = pd.DataFrame(self.val_set)
        train.to_csv(os.path.join(config.DATASET_OUTDIR, "Train.csv"), sep = ' ', header = False, index = False)
        test.to_csv(os.path.join(config.DATASET_OUTDIR, "Test.csv"), sep = ' ', header = False, index = False)
    