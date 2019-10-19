# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 05:02:41 2019

@author: Ashima
"""

import pickle
import pandas as pd
import config
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class Data:
    def __init__(self):
        self.dataset = None
        self.dataX = None
        self.dataY = None
        self.size = None 

    def preprocess(self):
        maxs = np.max(self.dataX)
        mins = np.min(self.dataX)
        self.dataX = (self.dataX - mins)/(maxs - mins)
    
    def visualize_data(self):
        column_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean']
        self.dataset = self.dataset.iloc[:, 2:12]
        self.dataset.columns = column_names
        print(self.dataset.shape)
        sns.pairplot(data = self.dataset)
        print("pairplot done")
        
    def read(self, filename):
        file = os.path.join(config.DATA_DIR, filename)
        print("File: ", file)
        with open(file, 'rb') as f:
            data = f.read()   
        data = data.decode().split()
        for i in range(np.shape(data)[0]):
            data[i] = data[i].split(',')   
        self.dataset = pd.DataFrame.from_records(data)
        self.dataY = self.dataset.iloc[:,1]
        self.dataX = self.dataset.drop(self.dataset.columns[[0, 1]], axis = 1)
        self.dataX = self.dataX.to_numpy()
        labelEncoder_Y = LabelEncoder()
        self.dataY = labelEncoder_Y.fit_transform(self.dataY)

    def get_data(self):
        return self.dataX, self.dataY
    