# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 05:27:49 2019

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
        gradient = (np.dot(np.transpose(X), t)) + (config.LAMBDA/(2*n))*(np.sign(self.W))
        gradient[0] = gradient[0] - (config.LAMBDA/(2*n))*(np.sign(self.W[0]))        
        self.W = self.W - (config.LEARNING_RATE/n)*(gradient)
    
    def lossFn(self, Y_train, Y_predict):
        Y_diff = Y_predict - Y_train
        n = Y_train.shape[0]
        loss = (1/(2*n))*(np.dot(np.transpose(Y_diff), Y_diff)) + (config.LAMBDA/(2*n))*(np.sum(np.abs(self.W)) - np.abs(self.W[0]))
        #loss = np.sqrt(loss[0][0])
        return loss
        
    def train(self, data):
        print("Start Training")
        X, Y = data.get_data()
        Y = Y.reshape(Y.shape[0], 1)
        self.n_examples = X.shape[0]
        X = np.c_[np.ones(self.n_examples), X]
        for epoch in range(config.NUM_EPOCHS):
            Y_predict = self.forward(X)
            loss = self.lossFn(Y, Y_predict)
            self.loss_lists.append(loss)
            if epoch % 100 == 0:
                print("Epoch: {}, Loss: {} ".format(epoch, loss))
            self.backward(X, Y, Y_predict)
        print("Final Loss: {}".format(loss))

    def test(self, data):
        print("Start testing")
        X, Y = data.get_data()
        Y = Y.reshape(Y.shape[0], 1)
        self.n_examples = X.shape[0]
        X = np.c_[np.ones(self.n_examples), X]
        Y_pred = self.forward(X)
        for i in range(10):
            print("Actual {}, Predicted {} ".format(Y[i], Y_pred[i]))
        
    def plot_best_fit_line(self, data):
        X, Y = data.get_data()
        plt.scatter(X, Y, color = 'r', marker = 'o', s = 3)
        Y_pred = self.W[0] + self.W[1] * X
        for i in range(10):
            print("Actual {}, Predicted {} ".format(Y[i], Y_pred[i]))
        plt.plot(X, Y_pred, color = 'g')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('L1 Regularization Using Gradient Descent')
        plt.savefig(config.OUT_DIR + 'BestFitLinePlot_L1Regularization_GradDescent')
        plt.show()