# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:57:09 2019

@author: Ashima
"""
import os
import time
import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class Model():
    
    def __init__(self):
        self.best_C = None
        self.best_gamma = None
        self.best_kernel = None
        self.best_estimator = None
        self.grid = None
        self.svm = None
    
    def gridsvc_train(self, data):
        params = { 'C':[0.1, 1, 10]}
        X_train, Y_train = data.get_data(False, True)
        X_train = X_train.reshape(X_train.shape[0], -1)
        start_time = time.time()
        model = SVC()
        self.grid = GridSearchCV(estimator = model, param_grid = params, cv = 2, n_jobs = -1)        #Default Value cv = 3
        self.grid.fit(X_train, Y_train)
        print("SVM best score ", self.grid.best_score_)
        print("Train Set Accuracy Score: {} ".format(self.grid.best_score_))
        self.best_C = self.grid.best_estimator_.C
        self.best_gamma = self.grid.best_estimator_.gamma
        self.best_kernel = self.grid.best_estimator_.kernel
        self.svm = self.grid.best_estimator_               #Set SVM Classifier required for next part to Best estimator obtained from GRID SEARCH
        self.best_params = self.grid.best_params_
        print("Best Params: ", self.best_params)
        print("Best Estimator: ", self.best_estimator)
        end_time = time.time()
        print("Model training time: {} sec".format(end_time - start_time))
       
    def svc_train(self, data, support):
        #self.svm = None
        #self.svm = SVC(**self.best_params)        #Working
        #self.svm = SVC(C = self.best_C, kernel = self.best_kernel, gamma = self.best_gamma)   
        X_train, Y_train = data.get_data(support, True)          #support is True when reading support vectors 
        X_train = X_train.reshape(X_train.shape[0], -1)
        self.svm.fit(X_train, Y_train)
        print("Train accuracy: ", self.svm.score(X_train, Y_train))
        if support == False:                                   #if not suppport vectors, save suppport vectors 
            support_vectors = self.svm.support_vectors_
            support_vector_indices = self.svm.support_
            print("Found Support Vectors: ", support_vector_indices.shape[0])
            print("Save Support Vectors")
            np.save(os.path.join(config.OUT_DIR, config.SUPPORT_LABELS), np.take(Y_train, support_vector_indices))
            np.save(os.path.join(config.OUT_DIR, config.SUPPORT_DATA), support_vectors)
        else:
            X_train, Y_train = data.get_data(False, True)        #Read Train Data , support = False, Train-true
            X_train = X_train.reshape(X_train.shape[0], -1)
            print("No of examples to train: ", X_train.shape[0])
            print("Train Accuracy On original Training Dataset: ", self.svm.score(X_train, Y_train)
            
    def svc_test(self, data, support):
        X_test, Y_test = data.get_data(support, False)
        X_test = X_test.reshape(X_test.shape[0], -1)
        print("Test accuracy: ", self.svm.score(X_test, Y_test))