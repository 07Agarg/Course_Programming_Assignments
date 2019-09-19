# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:06:32 2019

@author: ashima
"""

import pandas as pd
import config
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Data:
    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.size = None        
        #self.data_index = 0
    
    def preprocess(self):
        labelEncoder = LabelEncoder()
#        self.dataX[:, 1] = labelEncoder.fit_transform(self.dataX[:, 1])
#        self.dataX[:, 3] = labelEncoder.fit_transform(self.dataX[:, 3])
#        self.dataX[:, 5] = labelEncoder.fit_transform(self.dataX[:, 5])
#        self.dataX[:, 6] = labelEncoder.fit_transform(self.dataX[:, 6])
#        self.dataX[:, 7] = labelEncoder.fit_transform(self.dataX[:, 7])
#        self.dataX[:, 8] = labelEncoder.fit_transform(self.dataX[:, 8])
#        self.dataX[:, 9] = labelEncoder.fit_transform(self.dataX[:, 9])
#        self.dataX[:, 13] = labelEncoder.fit_transform(self.dataX[:, 13])
        self.dataY = labelEncoder.fit_transform(self.dataY)
        #self.dataX = self.dataX[:, [0, 2, 4, 10, 11, 12]]\
        #for i in range(2):
        #    print(self.dataX[:, i])
        maxs = np.max(self.dataX, axis = 0)
        mins = np.min(self.dataX, axis = 0)
        self.dataX = (self.dataX - mins)/(maxs - mins)
        
    def read(self, filename):
        data = pd.read_csv(os.path.join(config.DATA_DIR, filename) , sep = ', ', header = None)
        print("read successful")
        data = data[[0, 2, 4, 10, 11, 12, 14]]
        data = data.iloc[np.random.permutation(len(data))]
        data = np.asarray(data)
        self.dataX = data[:, :-1]   
        self.dataY = data[:, -1]
        self.preprocess()
        self.dataY = np.asarray(self.dataY, dtype = np.float).reshape(self.dataY.shape[0], 1)
        self.size = data.shape
        print("size ", self.size)
    
    def get_data(self):
        return self.dataX, self.dataY
    
    def get_test(self):
        return self.dataX, self.dataY