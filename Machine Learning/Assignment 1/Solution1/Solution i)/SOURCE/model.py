# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:31:48 2019

@author: ashima
"""
#Reference : Normal Eqn: http://cs229.stanford.edu/notes/cs229-notes1.pdf

import os
import config
import numpy as np
import matplotlib.pyplot as plt

class Model():
    
    def __init__(self):
        #self.W = np.random.randn(self.size).reshape(self.size, 1)
        self.n_examples = 0
        self.loss_lists = []                           #Stores Lists of loss Values in training set in 5 folds
        self.loss_lists_val = []                       #Stores Lists of loss values in validation set in 5 folds
        self.rmse_loss_grad = []                       #RMSE Loss using Gradient over all folds 
        self.rmse_loss_normal = []                     #RMSE Loss using Normal Eqn. over all folds 
        self.rmse_loss_normal_test = []
        self.rmse_loss_grad_test = []
    
    def reinit(self, size):
        self.size = size
        self.W = np.zeros((self.size, 1))
        self.W_Normal = np.zeros((self.size, 1))
        self.n_examples = 0
        self.loss_list_grad = []                       #List to store and plot loss values of train for current Fold
        self.loss_list_grad_val = []                   #List to store and plot loss values of validation set for current Fold
    
    def forward(self, X):
        Y = np.dot(X, self.W)
        return Y
    
    def backward(self, X, Y, Y_predict):
        t = Y_predict - Y
        gradient = (np.dot(np.transpose(X), t))
        self.W = self.W - (config.LEARNING_RATE/self.n_examples)*(gradient)
    
    def lossFn(self, Y_train, Y_predict):
        Y_diff = Y_predict - Y_train
        loss = (1/(2*self.n_examples))*(np.dot(np.transpose(Y_diff), Y_diff))
        Loss = np.sqrt(loss[0][0])
        return Loss

    def normalEqn(self, data):
        X_train, Y_train = data.get_train()
        X_train = X_train.astype(np.float32)
        X_transpose = np.transpose(X_train)
        self.n_examples = X_train.shape[0]
        X_train = np.c_[np.ones(self.n_examples), X_train]
        self.W_Normal = np.dot(np.dot(np.linalg.pinv(np.dot(X_transpose, X_train)), X_transpose), Y_train)
        Y_predict = np.dot(X_train, self.W_Normal)
        self.rmse_loss_normal.append(self.lossFn(Y_train, Y_predict))
        #with open(os.path.join(config.OUT_DIR, "W_Normal.txt") , "w") as file:
        #    file.write(str(self.W_Normal))
        
    def train(self, data):
        print("Start Training")
        X_train, Y_train = data.get_train()
        self.n_examples = X_train.shape[0]
        X_train = np.c_[np.ones(self.n_examples), X_train]
        
        X_val, Y_val = data.get_val()
        X_val = np.c_[np.ones(X_val.shape[0]), X_val]
        
        for epoch in range(config.NUM_EPOCHS):
            Y_predict = self.forward(X_train)
            loss = self.lossFn(Y_train, Y_predict)
            self.loss_list_grad.append(loss)
            print("Epoch: {}, Loss: {} ".format(epoch, loss))
            
            Y_pred_val = self.forward(X_val)
            loss_val = self.lossFn(Y_val, Y_pred_val)
            self.loss_list_grad_val.append(loss_val)
            
            self.backward(X_train, Y_train, Y_predict)
            
        print("Final Loss: {}".format(loss))
        self.rmse_loss_grad.append(loss)
        self.loss_lists.append(self.loss_list_grad)
        self.loss_lists_val.append(self.loss_list_grad_val)
        
    def test(self, data):
        print("Start testing")
        X_test, Y_test = data.get_val()
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        Y_predict = self.forward(X_test)
        loss = self.lossFn(Y_test, Y_predict)
        self.rmse_loss_grad_test.append(loss)
        
    def save_tofile(self):
        #Store RMSE Loss using Normal Equation in File
        with open(os.path.join(config.OUT_DIR, config.RMSE_NORMAL_FILE) , "w") as file:
            file.write(str(self.rmse_loss_normal))
        #Store RMSE Loss using Gradient Descent in File
        with open(os.path.join(config.OUT_DIR, config.RMSE_GRAD_FILE) , "w") as file:
            file.write(str(self.rmse_loss_grad))
        #Store RMSE Loss For Validation using Normal Equation in File
        with open(os.path.join(config.OUT_DIR, config.RMSE_VAL_NORMAL_FILE) , "w") as file:	
            file.write(str(self.rmse_loss_normal_test))
        #Store RMSE Loss For Validation using Gradient Descent in File
        with open(os.path.join(config.OUT_DIR, config.RMSE_VAL_GRAD_FILE) , "w") as file:	
            file.write(str(self.rmse_loss_grad_test))
        
    def plot_rmse(self, string):
        print("Plot RMSE")
        markers = ['.', 'o', 'v', '^', '<']
        colors = ['r', 'b', 'g', 'c', 'm']
        x = np.arange(config.NUM_EPOCHS)
        if string == "train":
            for i in range(config.K_FOLDS):
                plt.plot(np.asarray(x), np.asarray(self.loss_lists[i]), label = "Fold " + str(i), color = colors[i], marker = markers[i])
        else:
            for i in range(config.K_FOLDS):
                plt.plot(np.asarray(x), np.asarray(self.loss_lists_val[i]), label = "Fold " + str(i), color = colors[i], marker = markers[i])
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.title("RMSE Plot for {} Folds of {}".format(config.K_FOLDS, string))
        plt.legend()
        plt.savefig(config.OUT_DIR + 'RMSE for ' + str(string) + '.jpg')
        plt.show()
        
    def test_normalEqn(self, data):
        print("Start Testing Using Normal Eqn")
        X_test, Y_test = data.get_val()
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        Y_test_predict = np.dot(X_test, self.W_Normal)
        loss = self.lossFn(Y_test, Y_test_predict)
        print("normal val loss ", loss)
        self.rmse_loss_normal_test.append(loss)