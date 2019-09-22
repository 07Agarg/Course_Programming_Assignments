# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:14:00 2019

@author: ashima
"""


import config
import data
import model

if __name__ == "__main__":
    #print("Start")
    data = data.Data()
    #Read train 
    
    #L1 Regularization
    data.read(config.TRAIN_INPUT, config.TRAIN_LABELS, True)
    print("Train data read successfully")
    
    model = model.Model(data.size)
    model.LogisticReg_L1_train(data, 'saga')
    print("Training Using L1 Regularization")
    
    data.read(config.TEST_INPUT, config.TEST_LABELS, False)
    print("Test Data read Successfully")
    model.LogisticReg_L1_test(data)
    print("Testing Using L1 Regularization")
    
    
    """
    #L2 Regularization
    data.read(config.TRAIN_INPUT, config.TRAIN_LABELS, True)
    print("Train data read successfully")
    
    model = model.Model(data.size)
    model.LogisticReg_L2_train(data, 'lbfgs')
    print("Training Using L2 Regularization")
    
    data.read(config.TEST_INPUT, config.TEST_LABELS, False)
    print("Test Data read Successfully")
    model.LogisticReg_L2_test(data)
    print("Testing Using L2 Regularization")
    """
    """
    data.read(config.TRAIN_INPUT, config.TRAIN_LABELS, True)
    print("Train data read successfully")
    model = model.Model(data.size)
    model.Logistic_sgdclassifier_L1train(data)
    print("Train using sgdclassifier using L1 Regularization")
    """