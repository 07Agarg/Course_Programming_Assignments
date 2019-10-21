# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:57:09 2019

@author: Ashima
"""
import time
import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
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
        params = {'C':[0.1, 1, 10]}
        #'gamma':[0.1, 1, 10]}
        #'kernel': 'rbf'}
        X_train, Y_train = data.get_data()
        start_time = time.time()
        model = SVC()
        self.grid = GridSearchCV(estimator = model, param_grid = params, cv = 2, n_jobs = -1)        #Default Value cv = 3
        self.grid.fit(X_train, Y_train)
        print("SVM best score ", self.grid.best_score_)
        print("SVM best estimator ", self.grid.best_estimator_)
        print("Train Set Accuracy Score: {} ".format(self.grid.best_score_))
        self.best_C = self.grid.best_estimator_.C
        self.best_gamma = self.grid.best_estimator_.gamma
        self.best_kernel = self.grid.best_estimator_.kernel
        self.best_estimator = self.grid.best_estimator_
        end_time = time.time()
        print("Model training time: {} sec".format(end_time - start_time))
        
    def gridsvc_test(self, data):
        X_test, Y_test = data.get_data()
        Y_predict = self.grid.predict(X_test)
        print("Test Set Accuracy Score: {} ".format(accuracy_score(Y_test, Y_predict)))
        
    def svc_train(self, data):
        print("Best gamma: ", self.best_gamma)
        self.svm = SVC(C = self.best_C, kernel = self.best_kernel, gamma = self.best_gamma)
        #self.svc_train = SVC(self.best_estimator)
        X_train, Y_train = data.get_data()
        self.svm.fit(X_train, Y_train)
        support_vector_indices = self.svm.support_
        print(support_vector_indices.shape)
        
#    def save_tofile(self):
#        #Store Regularization Paramter in File
#        with open(os.path.join(config.OUT_DIR, config.REGULARIZATION_PARAM) , "w") as file:
#            file.write("Best Regularization Parameter for L2\n")
#            file.write(str(self.ridge_alpha))
#            file.write("\n\n\n")
#            file.write("Best Regularization Parameter for L1\n")
#            file.write(str(self.lasso_alpha))
#        