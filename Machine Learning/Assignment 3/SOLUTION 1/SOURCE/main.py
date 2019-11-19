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
    #data.read(config.TRAIN_INPUT, config.TRAIN_LABELS, True)
    X, Y = data.read(config.MNIST_FILE)
    print("Train data read successfully")
    
    X_, Y_ = X[:-config.NUM_TEST], Y[:-config.NUM_TEST]
    X_test, Y_test = X[config.NUM_TRAIN+config.NUM_VAL:], Y[config.NUM_TRAIN+config.NUM_VAL:]
    
    network = neural_network.Network(config.NUM_TRAIN, config.IMAGE_SIZE*config.IMAGE_SIZE)
    network.train(X_, Y_)    
 #    #print("Complete model training")
    print("Accuracy Plot for Train")
    network.plot_accuracy('Train')
    print("Error Plot for Train")
    network.plot_cost('Train')
    print("Accuracy Plot for Validation")
    network.plot_accuracy('Validation')
    print("Error Plot for Validation")
    network.plot_cost('Validation')
    print("Plot T-SNE")
    network.plot_tsne()
     
 #    #Test on Holdout Set
    #data.read(config.TEST_INPUT, config.TEST_LABELS, False)
    #X_test, Y_test = data.get_data()
     
    print("Read Test Data")
    print(X_test.shape)
    print(Y_test.shape)
    network.test(X_test.T, Y_test.T)
     
    #network.sklearn_train(X, Y)
