# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:43:55 2019

@author: Ashima
"""

import data
import config
import random
import utils
import numpy as np
import neural_network

if __name__ == "__main__":
    data = data.Data()
    X, Y = data.read(config.MNIST_FILE)
    
    Y = np.where(Y == 7, 0, 1)              ##Important to change Y as 0 where it is 7, Y as 1 where it is 9
    print("Train data read successfully")
    
    X_, Y_ = X[:-config.NUM_TEST], Y[:-config.NUM_TEST]
    X_test, Y_test = X[config.NUM_TRAIN+config.NUM_VAL:], Y[config.NUM_TRAIN+config.NUM_VAL:]
    
    config.CLASS_LABELS = np.unique(Y_)
    config.CLASSES = len(config.CLASS_LABELS)

    network = neural_network.Network(config.NUM_TRAIN, config.IMAGE_SIZE*config.IMAGE_SIZE, len(config.CLASS_LABELS))

    network.train(X_, Y_)    
 
    print("Accuracy Plot for Train")
    network.plot_accuracy('Train')
    print("Error Plot for Train")
    network.plot_cost('Train')
    print("Accuracy Plot for Validation")
    network.plot_accuracy('Validation')
    print("Error Plot for Validation")
    network.plot_cost('Validation')
    print("Plot T-SNE")
    utils.plot_tsne()
     
    print("Read Test Data")
    network.test(X_test.T, Y_test.T)

    #network.sklearn_train(X_, Y_)
    #network.sklearn_test(X_test, Y_test)
