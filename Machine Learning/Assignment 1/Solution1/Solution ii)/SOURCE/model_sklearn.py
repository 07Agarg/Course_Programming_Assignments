# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 04:38:54 2019

@author: ashima
"""

import os
import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

class Model():
    def __init__(self):
        self.ridge_alpha = None
        self.lasso_alpha = None
    
    def ridge_regression(self, data):
        X_train, Y_train = data.get_train()
        #alpha = regularization parameter
        params = {'alpha':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
        model = Ridge()
        grid = GridSearchCV(estimator = model, param_grid = params, cv = 5)
        grid.fit(X_train, Y_train)
        print("Ridge regression best score ",grid.best_score_)
        print("Ridge regression best alpha ", grid.best_estimator_.alpha)
        self.ridge_alpha = grid.best_estimator_.alpha
        
    def lasso_regression(self, data):
        X_train, Y_train = data.get_train()
        params = {'alpha':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
        model = Lasso()
        grid = GridSearchCV(estimator = model, param_grid = params, cv = 5)
        grid.fit(X_train, Y_train)
        print("Lasso regression best score ", grid.best_score_)
        print("Lasso regression best alpha ", grid.best_estimator_.alpha)
        self.lasso_alpha = grid.best_estimator_.alpha
    
    def save_tofile(self):
        #Store Regularization Paramter in File
        with open(os.path.join(config.OUT_DIR, config.REGULARIZATION_PARAM) , "w") as file:
            file.write("Best Regularization Parameter for L2\n")
            file.write(str(self.ridge_alpha))
            file.write("\n\n\n")
            file.write("Best Regularization Parameter for L1\n")
            file.write(str(self.lasso_alpha))
