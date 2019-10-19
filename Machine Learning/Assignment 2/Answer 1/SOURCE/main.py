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
    
    model = model.Model()
    model.gridsvc_train(data)
    print("Model trained")
#    data.read(config.TEST_FILE)
#    model.gridsvc_test(data)
#    print("model tested")
    
    #data.read(config.T)
    model.svc_train(data)
    print("Model trained with best estimators")
#    model.svc_test(data)
#    print("Model tested")
    
    
#    #Model object
#    model = model.Model()        
#    for i in range(config.K_FOLDS):
#        data.get_data(i)
#        #Reinitialize Parameters
#        model.reinit(data.size[1])
#        #Train data keeping ith fold as validation set
#        model.train(data)        
#        #print("Training Completed")
#        model.test(data)
#        #Using Normal Equation
#        model.normalEqn(data)
#        #RMSE on Validation Set Using Normal Equation
#        model.test_normalEqn(data)
#        
#    #Save RMSE to File
#    model.save_tofile()
#    #Plot RMSE for Training Data
#    model.plot_rmse("Train")      
#    #Plot RMSE for Validation Data
#    model.plot_rmse("Validation")
#    index = utils.find_min_rmse_val()
#    data.get_data(i)
#    data.save_data()
#
#    