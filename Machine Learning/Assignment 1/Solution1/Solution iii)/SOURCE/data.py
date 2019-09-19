# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:27:29 2019

@author: ashima
"""

import pandas as pd
import config
import os
import numpy as np

class DATA:
    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.size = 0
    
    def read(self, filename):
        data = pd.read_csv(os.path.join(config.DATA_DIR, filename) , sep = ',')
        self.dataX = data['Brain_Weight']
        self.dataY = data['Body_Weight']
        self.dataX = np.array(self.dataX)
        self.dataY = np.array(self.dataY)
        self.size = data.shape
        #print("sel.size ", self.size)
        print("read successful")
    
    def get_data(self):
        return self.dataX, self.dataY