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
        self.dataX_featured_train = None                       #Stores dataX when features extracted with HOG 
        self.dataX_featured_test = None
        self.dataY_train = None
        self.dataY_test = None
        self.dataY_support = None
        self.dataX_support = None
        
    def extract_hog_features(self, img):
        #print("img shape: ", img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_hog = hog(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), block_norm='L2-Hys', visualize=False, transform_sqrt=True, feature_vector=True, multichannel=None)
        #print("hog feature shape: ", img_hog.shape)
        #print(img_hog.shape)
        return img_hog
    
    def preprocess_data_hog(self, train):
        data_size = self.dataX.shape[0]
        print("dataX shape: ", self.dataX.shape)
        print("dataY shape: ", self.dataY.shape)
        self.dataX = np.reshape(self.dataX, (data_size, config.CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
        print("dataX reshaped shape: ", self.dataX.shape)
        self.dataX = np.array(self.dataX)
        self.dataX = self.dataX.transpose(0, 2, 3, 1)
        self.dataX_featured = np.zeros((data_size, config.HOG_FEATURES))
        start_time = time.time()
        for i in range(data_size):
            self.dataX_featured[i] = self.extract_hog_features(self.dataX[i])
        end_time = time.time()
        print("feature extraction time: {} sec".format(end_time - start_time))
        print(self.dataX_featured.shape)
        if train:
            self.dataX_featured_train = self.dataX_featured
            self.dataY_train = self.dataY
        else:
            self.dataX_featured_test = self.dataX_featured
            self.dataY_test = self.dataY
#        print("dataX reshaped shape now : ", self.dataX.shape)
#        cv2.imshow('first image', self.dataX[1])
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
    def preprocess_data_shrink(self, data_size, train):
        print("dataX shape: ", self.dataX.shape)
        print("dataY shape: ", self.dataY.shape)
        self.dataX = np.reshape(self.dataX, (data_size, config.CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
        print("dataX reshaped shape: ", self.dataX.shape)
        self.dataX = np.array(self.dataX)
        start_time = time.time()
        self.dataX = self.dataX.transpose(0, 2, 3, 1)
        self.dataX_featured = np.zeros((data_size, config.SHRINK_SIZE, config.SHRINK_SIZE))
        for i in range(data_size):
            img = self.dataX[i]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (config.SHRINK_SIZE, config.SHRINK_SIZE))
            self.dataX_featured[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
            #print("gray img shape: ", img.shape)
        end_time = time.time()
        print("feature extraction time: {} sec".format(end_time - start_time))
        print(self.dataX_featured.shape)
        if train:
            self.dataX_featured_train = self.dataX_featured
            self.dataY_train = self.dataY
        else:
            self.dataX_featured_test = self.dataX_featured
            self.dataY_test = self.dataY
    
    
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
            
    def read_support_vec(self, datafile, labelsfile):
        data = os.path.join(config.OUT_DIR, datafile)
        labels = os.path.join(config.OUT_DIR, labelsfile)
        self.dataY_support = np.load(labels+".npy")
        self.dataX_support = np.load(data+".npy")

    def get_data(self, support, train):
        if support == False:
            if train:
                return self.dataX_featured_train, self.dataY_train
            else:
                return self.dataX_featured_test, self.dataY_test
        return self.dataX_support, self.dataY_support
    
    def save_data(self):
        train = pd.DataFrame(self.train_set)
        test = pd.DataFrame(self.val_set)
        train.to_csv(os.path.join(config.DATASET_OUTDIR, "Train.csv"), sep = ' ', header = False, index = False)
        test.to_csv(os.path.join(config.DATASET_OUTDIR, "Test.csv"), sep = ' ', header = False, index = False)
    