# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:05:14 2019

@author: ashima
"""
import config
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv(os.path.join(config.DATA_DIR, config.TRAIN_PATH) , sep = ', ', header = None)
print("read successful")
data = data[[0, 2, 4, 10, 11, 12, 14]]
data = data.iloc[np.random.permutation(len(data))]
data = np.asarray(data)
dataX = data[:, :-1]   
dataY = data[:, -1]

labelEncoder = LabelEncoder()
dataY = labelEncoder.fit_transform(dataY)
maxs = np.max(dataX, axis = 0)
mins = np.min(dataX, axis = 0)
dataX = (dataX - mins)/(maxs - mins)
dataY = np.asarray(dataY, dtype = np.float) #.reshape(dataY.shape[0], 1)
lr = LogisticRegression(penalty = 'l1', max_iter = 100)   #default C = 1.0 which is inverse of regualrization strnegth
X_train, y_train = dataX, dataY
classifier = lr.fit(X_train, y_train)
y_predict = classifier.predict(X_train)
#y1 = classifier.predict_proba(X_train)
#cm = confusion_matrix(y_train, y_predict)
print("Confusion Matrix For Train set")
#class_accuracy(cm, "Train")
train_accuracy = classifier.score(X_train, y_train)
        