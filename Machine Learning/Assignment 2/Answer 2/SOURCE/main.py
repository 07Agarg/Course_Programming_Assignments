# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 05:02:08 2019

@author: Ashima
"""

import config
import data
import model

if __name__ == "__main__":
    data = data.Data()
    #Read Train Data and load Train data
    train_data = data.read(config.FILE3)
    train = train_data.decode()
    #train[0] = train[0].split(',')
    #print(type(train_data))
#    data.read(config.LABELS_FILE_PATH)
#    print("data read successfully")
#    
#    model = model.Model()
#    #Train using SVM CLassifier
#    model.svc_train(data)
#    print("Model trained")
#    #Train using Gaussian Naive Bayes Classifier
#    model.naive_bayes_train(data)
#    #Train using Decision trees Classifier
#    model.decision_trees_train(data)
#    
#    #Read Test File
#    data.read(config.TEST_FILE)
#    #Test using SVC
#    model.svc_test(data)
#    #Test using Naive Bayes
#    model.naive_bayes_test(data)
#    #Test using Decision Trees Classifier
#    model.decision_trees_test(data)