# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 03:36:55 2019

@author: Ashima
"""

import data
import config
import model

if __name__ == "__main__":
    #print("Start")
    data = data.Data()
    #Read train data

#    train_dataloader = data.read(config.TRAIN_FILE)
#    print("Train data read successfully")
#    net = models.AlexNet()
#    print("Printing Alexnet Arch.... ")
#    print(net)
#    #Extract and save features
#    Y = data.extract_features(net, config.TRAIN_FEATURES)
#    print("Extracted features for train set")
#    #Read test data
#    
#    test_data = data.read(config.TEST_FILE)
#    print("Test data read successfully")
#    data.extract_features(net, config.TEST_FEATURES)
#    print("Extracted features for test set")        

    data.read_features(config.TRAIN_FEATURES)
    print("Read Train Features Done")
    model = model.Model()
    model.svc_train(data)
    print("Model Trained")
    
    data.read_features(config.TEST_FEATURES)
    print("Read Test Features Done")
    model.svc_test(data)
    print("Model Tested")
    
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
