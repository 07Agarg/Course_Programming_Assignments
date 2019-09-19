# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:00:59 2019

@author: ashima
"""

import config
import data
import model
import utils

if __name__ == "__main__":
    data = data.Data()
    #Read Data and load Train data
    data.read(config.FILE_PATH)
    print("data read successfully")
    #Cross Validation Data using Gradient Descent     
    #Model object
    model = model.Model()        
    for i in range(config.K_FOLDS):
        data.get_data(i)
        #Reinitialize Parameters
        model.reinit(data.size[1])
        #Train data keeping ith fold as validation set
        model.train(data)        
        #print("Training Completed")
        model.test(data)
        #Using Normal Equation
        model.normalEqn(data)
        #RMSE on Validation Set Using Normal Equation
        model.test_normalEqn(data)
        
    #Save RMSE to File
    model.save_tofile()
    #Plot RMSE for Training Data
    model.plot_rmse("Train")      
    #Plot RMSE for Validation Data
    model.plot_rmse("Validation")
    index = utils.find_min_rmse_val()
    data.get_data(i)
    data.save_data()

    