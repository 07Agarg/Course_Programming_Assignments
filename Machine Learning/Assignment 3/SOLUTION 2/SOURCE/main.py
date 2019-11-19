# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 03:36:55 2019

@author: Ashima
"""

import data
import config
import model
import random
import numpy as np
from torch.autograd import Variable
from torchvision import models

if __name__ == "__main__":
    #print("Start")
    data = data.Data()
    #Read train data
# =============================================================================
# 
#     train_dataloader = data.read(config.TRAIN_FILE)
#     print("Train data read successfully")
#     net = models.AlexNet()
#     #print("Printing Alexnet Arch.... ")
#     #print(net)
#     #Extract and save features
#     Y = data.extract_features(net, config.TRAIN_FEATURES)
#     print("Extracted features for train set")
#     #Read test data
#     
#     test_data = data.read(config.TEST_FILE)
#     print("Test data read successfully")
#     data.extract_features(net, config.TEST_FEATURES)
#     print("Extracted features for test set")
#     
# =============================================================================
    data.read_features(config.TRAIN_FEATURES)
    print("Read Train Features Done")
    model = model.Model()
    model.svc_train(data)
    print("Model Trained")
    
    data.read_features(config.TEST_FEATURES)
    print("Read Test Features Done")
    model.svc_test(data)
    print("Model Tested")
    
# =============================================================================
#     
#     model_conv = models.alexnet(pretrained=True)
#     feature_extractor = model.FeatureExtractor(model_conv)
#     #print(model_conv)
#     print("Printing Feature Extractor....")
#     print(feature_extractor)
# =============================================================================
    
#    for data in train_dataloader:
#        X, Y = data
#        Y1 = net.forward(X).tolist()
#        print(np.shape(Y.tolist()))
#        print(np.shape(Y1))
    
#    features = Variable(feature_extractor(high_res_real).data)
   
    
# =============================================================================
#     
#     X, Y = data.get_data()
#     print("Read Train and Validation Data")
# #    #Split the data into training and validation set
#     X_train, Y_train = X[:config.NUM_TRAIN], Y[:config.NUM_TRAIN]
#     X_val, Y_val = X[config.NUM_TRAIN:], Y[config.NUM_TRAIN:]
# #    
#     print(X_train.shape)
#     print(Y_train.shape)
#     print(X_val.shape)
#     print(Y_val.shape)
# #    
#     network = neural_network.Network(X_train.shape[0], X_train.shape[1])
#     network.train(X_train.T, Y_train.T, True)    
# #    #print("Complete model training")
#     print("Plot Accuracy")
#     network.plot_accuracy('Train')
#     print("Plot Error")
#     network.plot_cost('Train')
# #    #network.train(X_val.T, Y_val.T, False)
# #    
# #    #Test on Holdout Set
#     data.read(config.TEST_INPUT, config.TEST_LABELS, False)
#     X_test, Y_test = data.get_data()
#     print("Read Test Data")
#     print(X_test.shape)
#     print(Y_test.shape)
#     network.test(X_test.T, Y_test.T)
#     
#     
#     network.sklearn_train(X, Y)
# =============================================================================
