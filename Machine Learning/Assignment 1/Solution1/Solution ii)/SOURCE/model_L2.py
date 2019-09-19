# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 05:14:07 2019

@author: ashima
"""

import config
import numpy as np
import matplotlib.pyplot as plt

class Model():
    
    def __init__(self, size):
        #self.W = np.random.randn(self.size).reshape(self.size, 1)
        self.n_examples = 0
        self.loss_lists = []
        self.size = size
        self.W = np.zeros((self.size, 1))
         
    def forward(self, X):
        Y = np.dot(X, self.W)
        return Y
    
    def backward(self, X, Y, Y_predict):
        t = Y_predict - Y
        n = Y.shape[0]        
        gradient = (np.dot(np.transpose(X), t)) + (config.LAMBDA_L2/n)*(self.W)
        gradient[0] = gradient[0] - ((config.LAMBDA_L2/n)*gradient[0])
        self.W = self.W - (config.LEARNING_RATE/n)*(gradient)
    
    def lossFn(self, Y_train, Y_predict):
        Y_diff = Y_predict - Y_train
        n = Y_train.shape[0]        
        loss = (1/(2*n))*(np.dot(np.transpose(Y_diff), Y_diff)) + (config.LAMBDA_L2/(2*n))*(np.dot(self.W.T, self.W) - (self.W[0]*self.W[0]))
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
            self.loss_lists.append(loss)
            if epoch % 100 == 0:
                print("Epoch: {}, Loss: {} ".format(epoch, loss))
            self.backward(X_train, Y_train, Y_predict)
        print("Final Loss: {}".format(loss))
        #with open(os.path.join(config.OUT_DIR, "W_Grad.txt") , "w") as file:
        #    file.write(str(self.W))
        
    def test(self, data):
        print("Start testing")
        X_test, Y_test = data.get_test()
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        Y_test_predict = self.forward(X_test)
        loss = self.lossFn(Y_test, Y_test_predict)
        print("Test Set RMSE Loss using L2 Regularization {} ".format(loss))
        
    def plot_rmse(self, string):
        print("Plot RMSE")
        print("Plot RMSE")
        x = np.arange(config.NUM_EPOCHS)
        plt.plot(np.asarray(x), np.asarray(self.loss_lists))
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.title("RMSE Plot for {} ".format(string))
        plt.legend()
        plt.savefig(config.OUT_DIR + 'RMSE_' + str(string) + '.jpg')
        plt.show()
        
        
#        for i in range(10):
#        #for i in range(X_test.shape[0]):
#            Y_test_predict = self.forward(X_test)
#            loss = self.lossFn(Y_test, Y_test_predict)
#            print("Test Sample No: {}, Actual Value: {}, Predicted_Value: {} Loss: {}".format(i, Y_test[i], Y_test_predict[i], loss))
#              