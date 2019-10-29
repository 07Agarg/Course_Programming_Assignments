# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:57:09 2019

@author: Ashima
"""

import config
import data
import model

if __name__ == "__main__":
    data = data.Data()
    #Read Train Data and load Train data
    data.read(config.TRAIN_FILE1)
    data.read(config.TRAIN_FILE2)
    data.read(config.TRAIN_FILE3)
    data.read(config.TRAIN_FILE4)
    data.read(config.TRAIN_FILE5)
    print("data read successfully")
    
    data.preprocess_data_hog(True)                  #Pass True when loading train data
    print("preprocessing done using HOG")
    
    #data.preprocess_data_shrink(config.TRAIN_SIZE, True)
    #print("preprocessing done without HOG")
    
    model = model.Model()
    
    model.gridsvc_train(data)
    print("grid search done")    
    
    model.svc_train(data, False)                   #Second argument False implies Not using Support Vectors       
    print("Model trained with best estimators")
    
    #Read Test Data
    data.read_test(config.TEST_FILE)
    print("test data read successfully") 
    data.preprocess_data_hog(False)                #Pass False when loading test data
    print("preprocessing done of testing data")
    #Model test on testing data
    model.svc_test(data, False)
    print("Model tested")
    
    #Load Support Vectors
    data.read_support_vec(config.SUPPORT_DATA, config.SUPPORT_LABELS)
    print("Read Support Vectors")
    #Train on support vectors
    model.svc_train(data, True)
    print("Model trained on support Vectors and tested on original training data ")
    
    #Test on support vectors
    model.svc_test(data, False)
    print("Model(trained on support vectors) tested on original test data")
    