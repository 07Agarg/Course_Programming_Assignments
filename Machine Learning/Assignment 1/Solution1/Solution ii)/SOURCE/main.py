# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:00:59 2019

@author: ashima
"""

import config
import data
import model_sklearn
import model_L1
import model_L2
#import utils

if __name__ == "__main__":
    data = data.Data()
    #Read Data and load Train data
    data.read(config.TRAIN_PATH)
    print("data read successfully")
    """
    #GridSearchcv for Optimal Hyperparameter 
    #Model object
    model = model_sklearn.Model()        
    model.ridge_regression(data)
    model.lasso_regression(data)
    #Save Best ALphas to file
    model.save_tofile()
    
    #L1 Regularization
    model_l1 = model_L1.Model(data.size[1])
    data.read(config.TRAIN_PATH)
    model_l1.train(data)
    model_l1.test(data)
    model_l1.plot_rmse("L1 Regularization")
    """
    #L2 Regularization
    model_l2 = model_L2.Model(data.size[1])
    data.read(config.TRAIN_PATH)
    model_l2.train(data)
    model_l2.test(data)
    model_l2.plot_rmse("L2 Regularization")
    #"""