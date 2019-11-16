# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:43:55 2019

@author: Ashima
"""

import config
import data
import numpy as np
import neural_network

if __name__ == "__main__":
    #print("Start")
    data = data.Data()
    data.read(config.TRAIN_INPUT, config.TRAIN_LABELS, True)
    print("Train data read successfully")
    
    X_train, Y_train = data.get_train()
    print(X_train.shape)
    print(Y_train.shape)
    network = neural_network.Network(X_train.shape[0], X_train.shape[1])
    network.model(X_train.T, Y_train.T)    