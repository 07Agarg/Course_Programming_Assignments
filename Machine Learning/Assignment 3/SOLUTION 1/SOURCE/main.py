# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:43:55 2019

@author: Ashima
"""

import data
import config
import random
import numpy as np
import neural_network

if __name__ == "__main__":
    #print("Start")
    data = data.Data()
    data.read(config.TRAIN_INPUT, config.TRAIN_LABELS, True)
    print("Train data read successfully")
    
    X, Y = data.get_data()
    print("Read Train and Validation Data")
    #Split the data into training and validation set
    X_train, Y_train = X[:config.NUM_TRAIN], Y[:config.NUM_TRAIN]
    X_val, Y_val = X[config.NUM_TRAIN:], Y[config.NUM_TRAIN:]
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)
    
    network = neural_network.Network(X_train.shape[0], X_train.shape[1])
    network.train(X_train.T, Y_train.T, True)    
    #print("Complete model training")
    print("Plot Accuracy")
    network.plot_accuracy()
    print("Plot Error")
    network.plot_cost()
    #network.train(X_val.T, Y_val.T, False)
    
    #Test on Holdout Set
    data.read(config.TEST_INPUT, config.TEST_LABELS, False)
    X_test, Y_test = data.get_data()
    print("Read Test Data")
    print(X_test.shape)
    print(Y_test.shape)
    network.test(X_test.T, Y_test.T)