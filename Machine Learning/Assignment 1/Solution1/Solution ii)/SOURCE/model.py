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
        self.loss_lists = []
        self.rmse_loss_grad = []                       #RMSE Loss using Gradient over all folds 
        self.rmse_loss_normal = []                     #RMSE Loss using Normal Eqn. over all folds 
        self.rmse_loss_normal_test = []
    
    def reinit(self, size):
        self.size = size
        self.W = np.zeros((self.size, 1))
        self.W_Normal = np.zeros((self.size, 1))
        self.n_examples = 0
        self.loss_list_grad = []                       #List to store and plot loss values for current Fold
    
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
        
    def train(self, data):
        print("Start Training")
        X_train, Y_train = data.get_train()
        self.n_examples = X_train.shape[0]
        X_train = np.c_[np.ones(self.n_examples), X_train]
        for epoch in range(config.NUM_EPOCHS):
            Y_predict = self.forward(X_train)
            loss = self.lossFn(Y_train, Y_predict)
            self.loss_list_grad.append(loss)
            if epoch % 100 == 0:
                print("Epoch: {}, Loss: {} ".format(epoch, loss))
            self.backward(X_train, Y_train, Y_predict)
        print("Final Loss: {}".format(loss))
        #with open(os.path.join(config.OUT_DIR, "W_Grad.txt") , "w") as file:
        #    file.write(str(self.W))
        self.rmse_loss_grad.append(loss)
        self.loss_lists.append(self.loss_list_grad)
        
    def test(self, data):
        print("Start testing")
        X_test, Y_test = data.get_val()
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        for i in range(10):
        #for i in range(X_test.shape[0]):
            Y_test_predict = self.forward(X_test)
            loss = self.lossFn(Y_test, Y_test_predict)
            print("Test Sample No: {}, Actual Value: {}, Predicted_Value: {} Loss: {}".format(i, Y_test[i], Y_test_predict[i], loss))
    
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
            
        
    def plot_rmse(self, string):
        print("Plot RMSE")
        markers = ['.', 'o', 'v', '^', '<']
        colors = ['r', 'b', 'g', 'c', 'm']
        x = np.arange(config.NUM_EPOCHS)
        for i in range(config.K_FOLDS):
            plt.plot(np.asarray(x), np.asarray(self.loss_lists[i]), label = "Fold " + str(i), color = colors[i], marker = markers[i])
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.title("RMSE Plot for {} Folds of {}".format(config.K_FOLDS, string))
        plt.legend()
        plt.savefig(config.OUT_DIR + 'RMSE for ' + str(string) + str(i) + '.jpg')
        plt.show()
        