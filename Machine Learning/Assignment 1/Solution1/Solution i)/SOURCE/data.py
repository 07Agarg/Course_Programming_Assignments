# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:06:32 2019

@author: ashima
"""

import pandas as pd
import config
import os
import numpy as np

class Data:
    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.size = None 
        self.folds = None
        self.val_set = None
        self.train_set = None
        #self.data_index = 0
        
    def preprocess(self, data):
        self.dataX = data[:, :-1]   
        self.dataX = data[:, 1:]            #Removing first feature
        self.dataY = data[:, -1]
        #print(self.dataY.shape)
        self.dataY = np.asarray(self.dataY, dtype = np.float).reshape(self.dataY.shape[0], 1)
        self.dataY = self.dataY + 1.5        #Age = Rings + 1.5
        self.size = data.shape
        
    def generate_folds(self, data):
        extra = self.size[0] % config.K_FOLDS
        remain_data = data[-extra:]
        new_len = self.size[0]-extra
        new_data = data[:new_len]
        folds = np.split(new_data, config.K_FOLDS)
        for i in range(extra):
            folds[i] = np.vstack((folds[i], remain_data[i]))
        self.folds = folds
        #print("no of folds ", np.shape(folds))
        
    def read(self, filename):
        data = pd.read_csv(os.path.join(config.DATA_DIR, filename) , sep = ' ', header = None)  #names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
        #data = data.iloc[np.random.permutation(len(data))]
        self.size = data.shape
        data = np.asarray(data)
        self.generate_folds(data)
    
    def get_data(self, i):
        self.val_set = self.folds[i]
        self.train_set = np.array([]).reshape(0, self.folds[i].shape[1])
        for j in range(config.K_FOLDS):
            if j != i:
                self.train_set = np.vstack((self.train_set, self.folds[j]))        
        self.preprocess(self.val_set)
    
    def save_data(self):
        train = pd.DataFrame(self.train_set)
        test = pd.DataFrame(self.val_set)
        train.to_csv(os.path.join(config.DATASET_OUTDIR, "Train.csv"), sep = ' ', header = False, index = False)
        test.to_csv(os.path.join(config.DATASET_OUTDIR, "Test.csv"), sep = ' ', header = False, index = False)
    
    def get_train(self):
        self.preprocess(self.train_set)
        return self.dataX, self.dataY
    
    def get_val(self):
        self.preprocess(self.val_set)
        return self.dataX, self.dataY