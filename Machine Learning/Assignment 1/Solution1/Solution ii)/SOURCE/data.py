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
        
    def preprocess(self, data):
        self.dataX = data[:, :-1]   
        self.dataX = data[:, 1:]            #Removing first feature
        self.dataY = data[:, -1]
        print(self.dataY.shape)
        self.dataY = np.asarray(self.dataY, dtype = np.float).reshape(self.dataY.shape[0], 1)
        self.dataY = self.dataY + 1.5        #Age = Rings + 1.5
        self.size = data.shape
        
    def read(self, filename):
        data = pd.read_csv(os.path.join(config.DATA_DIR, filename) , sep = ' ', header = None)  #names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
        data = data.iloc[np.random.permutation(len(data))]
        self.size = data.shape
        data = np.asarray(data)
        self.preprocess(data)
        #self.generate_folds(data)
    
    def get_train(self):
        return self.dataX, self.dataY
    
    def get_test(self):
        return self.dataX, self.dataY