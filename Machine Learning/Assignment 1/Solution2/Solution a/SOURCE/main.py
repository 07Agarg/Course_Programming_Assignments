    # -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:00:59 2019

@author: ashima
"""

import config
import data
import model
import model_L1
import model_L2


if __name__ == "__main__":
    data = data.Data()
    #Read train csv file
    data.read(config.TRAIN_PATH)
    print("train data read successfully")
    
    """
    #Without Regularization
    model = model.Model(data.size[1])
    model.train(data)
    print("Training completed without Regularization")
    print("Plot Accuracy")
    model.plot_accuracy()
    print("Plot Error")
    model.plot_error()
    #Read test csv file
    data.read(config.TEST_PATH)
    print("Test Data Read Successfully")
    model.test(data)
    print("Predicted test values!!!!")
    
    
    #L1 Regularization
    model_l1 = model_L1.Model(data.size[1])
    model_l1.train(data)
    print("Training Completed using L1 Regularization")
    print("Plot Accuracy")
    model_l1.plot_accuracy("L1 Regularization")
    print("Plot Error")
    model_l1.plot_error("L1 Regularization")
    #Read test csv file
    data.read(config.TEST_PATH)
    print("Test Data Read Successfully")
    model_l1.test(data)
    print("Predicted test values using L1 Regularization!!!!")
    
    """
    #L2 Regularization
    model_l2 = model_L2.Model(data.size[1])
    model_l2.train(data)
    print("Training Completed using L2 Regularization")
    print("Plot Accuracy")
    model_l2.plot_accuracy("L2 Regularization")
    print("Plot Error")
    model_l2.plot_error("L2 Regularization")
    #Read test csv file
    data.read(config.TEST_PATH)
    print("Test Data Read Successfully")
    model_l2.test(data)
    print("Predicted test values using L2 Regularization!!!!")
    