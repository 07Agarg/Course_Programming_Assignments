# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:50:06 2019

@author: Ashima
"""

import os
import h5py
import config
import numpy as np
import gzip
from sklearn.preprocessing import StandardScaler

class Data:
    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.size = 0
    
    def preprocess(self):
        for i in range(2):
            print(self.dataX[:, i])
        maxs = np.max(self.dataX, axis = 0)
        mins = np.min(self.dataX, axis = 0)
        self.dataX = (self.dataX - mins)/(maxs - mins)
    
    def read_buffer(self, f_input, f_labels, num, train):
        #Read input
        f_input.read(16)
        buf = f_input.read(num * config.IMAGE_SIZE * config.IMAGE_SIZE * 1)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
        self.dataX = data
        self.size = self.dataX.shape[0]
        
        #print("Size f_input ", self.dataX.shape)
        #Read labels            
        f_labels.read(8)
        buf = f_labels.read(num)
        labels = np.frombuffer(buf, dtype = np.uint8).astype(np.int64)
        labels = labels.reshape(num, 1)
        self.dataY = labels
        scaler = StandardScaler()
        self.dataX = self.dataX.reshape(self.dataX.shape[0], -1)        
        self.dataX = scaler.fit_transform(self.dataX)
    
    def read_full(self, inputs, labels, train = True):
        f_input = gzip.open(os.path.join(config.DATA_DIR, inputs),'r')
        f_labels = gzip.open(os.path.join(config.DATA_DIR, labels),'r')

        if train:
            self.read_buffer(f_input, f_labels, config.NUM_SAMPLES, train)
            
        if not train:
            self.read_buffer(f_input, f_labels, config.NUM_TEST, train)
        
    def read(self, input_file):
        filename = os.path.join(config.DATA_DIR, config.MNIST_FILE)
        data = h5py.File(filename, 'r+') 
        print(np.shape(data))
        X = data['X'][:]
        Y = data['Y'][:]
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], 1)
        X = X/255.
        print(X.shape)
        print(Y.shape)
        return X, Y
            
    def get_data(self):
        #return self.dataX[:10000], self.dataY[:10000]
        return self.dataX, self.dataY
    