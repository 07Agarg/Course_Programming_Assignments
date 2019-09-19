# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:12:09 2019

@author: ashima
"""

import config
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv(os.path.join(config.DATA_DIR, config.TEST_PATH) , sep = ', ', header = None)
le = LabelEncoder()
#print(type(file))
#a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#print(a.shape)
#a = np.squeeze(a) 

a = np.array([-0.1, 0.2, 0.3, -0.4, 0.5, 0.6])
a = np.abs(a)


"""

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mins = np.min(a, axis = 0)
maxs = np.max(a, axis = 0)
rng = maxs - mins
b = a
norm_X = 1 - ((maxs - b)/rng) 
b = a
norm_X1 = (b - mins)/rng
norm_X2 = a
for i in range(3):
    norm_X2[:, i] = norm_X2[:, i]/np.linalg.norm(norm_X2)
"""
"""
W = np.random.randn(9).reshape(9, 1)
a = ['M', 'F', 'F', 'M', 'F', 'F', 'M', 'I', 'I', 'I']
a = le.fit_transform(a)
"""
#b = np.array([1, 2, 3, 4]).reshape(4, 1)
#b = b + 1
"""
data = np.asarray(data)
dataX = data[:, :-1]   
dataY = data[:, -1]

labelEncoder = LabelEncoder()
dataX[:, 1] = labelEncoder.fit_transform(dataX[:, 1]) 
dataX[:, 2] = dataX[:,2] / np.linalg.norm(dataX[:,2])
dataX[:, 3] = labelEncoder.fit_transform(dataX[:, 3])
dataX[:, 5] = labelEncoder.fit_transform(dataX[:, 5])
dataX[:, 6] = labelEncoder.fit_transform(dataX[:, 6])
dataX[:, 7] = labelEncoder.fit_transform(dataX[:, 7])
dataX[:, 8] = labelEncoder.fit_transform(dataX[:, 8])
dataX[:, 9] = labelEncoder.fit_transform(dataX[:, 9])
dataX[:, 12] = dataX[:, 12]/np.linalg.norm(dataX[:, 12])
dataX[:, 13] = labelEncoder.fit_transform(dataX[:, 13])
dataY = labelEncoder.fit_transform(dataY)

for i in range(2):
    print(dataX[0], "  ", dataY[0])    
"""    