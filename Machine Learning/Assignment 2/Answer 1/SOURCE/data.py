# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:57:09 2019

@author: ashima
"""
import pickle
import pandas as pd
import config
import os
import numpy as np

class Data:
    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.size = None 

    def preprocess(self):
        maxs = np.max(self.dataX)
        mins = np.min(self.dataX)
        self.dataX = (self.dataX - mins)/(maxs - mins)
        
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
                #self.dataY = np.vstack((self.dataY, dataY))
                self.dataY = np.append(self.dataY, dataY)
                print(self.dataY.shape)

    def get_data(self):
        self.preprocess()
        return self.dataX[:100], self.dataY[:100]
    
    def save_data(self):
        train = pd.DataFrame(self.train_set)
        test = pd.DataFrame(self.val_set)
        train.to_csv(os.path.join(config.DATASET_OUTDIR, "Train.csv"), sep = ' ', header = False, index = False)
        test.to_csv(os.path.join(config.DATASET_OUTDIR, "Test.csv"), sep = ' ', header = False, index = False)
    