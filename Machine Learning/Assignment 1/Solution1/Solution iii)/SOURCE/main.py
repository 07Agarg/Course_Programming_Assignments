# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:28:18 2019

@author: ashima
"""
#Reference: https://www.emathzone.com/tutorials/basic-statistics/curve-fitting-and-method-of-least-squares.html
import config
import data
import model
import model_L1
import model_L2

if __name__ == "__main__":
    data = data.DATA()
    data.read(config.FILE_PATH)
    print("Read Data Successful")
    
    #Gradient Descent Model
    #Without Regularization
    model = model.Model(data.size[1])    
    #Train the model
    model.train(data)    
    #Plot Best Fit Line without Regularization
    model.plot_best_fit_line(data)    
    #Test the model after convergence
    model.test(data)
    
    #L1 Regularization
    model_l1 = model_L1.Model(data.size[1])
    #Train the model
    model_l1.train(data)
    #Plot best fit line using L1 regularization
    model_l1.plot_best_fit_line(data)
    #Test the model after convergence
    model_l1.test(data)
    #"""
    #L2 Regularization
    model_l2 = model_L2.Model(data.size[1])
    #Train the model
    model_l2.train(data)
    #PLot best fit line using L2 Regularization
    model_l2.plot_best_fit_line(data)
    #Test the model after convergence
    model_l2.test(data)
    #"""