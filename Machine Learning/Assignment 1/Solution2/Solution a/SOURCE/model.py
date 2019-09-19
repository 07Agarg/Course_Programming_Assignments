# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:31:48 2019

@author: ashima
"""
import os
import config
import numpy as np
import matplotlib.pyplot as plt


class Model():
    
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((self.size, 1))
        self.n_examples = 0
        self.loss_list = []
        self.accuracy_list = []
    
    def sigmoid(self, Z):
        p = 1.0/(1.0 + np.exp(-1.0 * Z))
        return p
    
    def forward(self, X):
        Z = np.array(np.dot(X, self.W), dtype = np.float32)
        return self.sigmoid(Z)
    
    def backward(self, X, Y, p):
        t = p - Y
        gradient = (np.dot(np.transpose(X), t))
        n = Y.shape[0]
        #print(gradient)
        self.W = self.W - (config.LEARNING_RATE/n)*(gradient)
    
    def lossFn(self, Y, p):
        loss = (-Y * np.log(p) - (1 - Y) * np.log(1 - p)).mean()
        #loss = (-1.0/self.n_examples)*(np.dot(np.transpose(Y), np.log(p)) + np.dot(np.transpose(1 - Y), np.log(1 - p)))
        return loss
    
    def train(self, data):
        print("Start Training")
        X, Y = data.get_data()        
        self.n_examples = X.shape[0]        
        #train validation split
        val_size = int(0.30*self.n_examples)
        train_size = self.n_examples - val_size
        X_train, Y_train = X[:train_size], Y[:train_size]
        X_val, Y_val = X[train_size:], Y[train_size:]
        print("Total Samples ", self.n_examples)
        X_train = np.c_[np.ones(train_size), X_train]
        for epoch in range(config.NUM_EPOCHS):
            p = self.forward(X_train)
            loss = self.lossFn(Y_train, p)
            print("Epoch: {}, Loss: {} ".format(epoch, loss))
            self.backward(X_train, Y_train, p)
            self.loss_list.append(loss)
            self.calculate_accuracy(X_train, Y_train, "None")
        print("Final Loss: {}".format(loss))
        with open(os.path.join(config.OUT_DIR, config.NOREG_WEIGHTS) , "w") as file:
            file.write(str(self.W))
        self.calculate_accuracy(X_train, Y_train, "Train")
        X_val = np.c_[np.ones(val_size), X_val]
        self.calculate_accuracy(X_val, Y_val, "Validation")
        
    def test(self, data):
        print("Start testing")
        X_test, Y_test = data.get_test()
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        self.calculate_accuracy(X_test, Y_test, "Test")
        
    def calculate_accuracy(self, X, Y, string):
        accuracy = 0
        for i in range(X.shape[0]):
            Y_pred = self.forward(X[i])
            if Y_pred >= 0.5:
                Y_pred_ = 1
            else:
                Y_pred_ = 0
            accuracy += (Y_pred_ == Y[i])
        if string == "None":
            self.accuracy_list.append((accuracy/Y.shape[0])*100)
        else:
            print("{} Set Accuracy {} without Regularization".format(string, (accuracy/Y.shape[0])*100))
    
    def plot_accuracy(self):
        x = np.arange(config.NUM_EPOCHS)
        plt.plot(np.asarray(x), np.asarray(self.accuracy_list))
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.title('Accuracy vs Epochs Plot')
        plt.savefig(config.OUT_DIR + 'Accuracy_Plot.jpg')
        plt.show()
    
    def plot_error(self):
        x = np.arange(config.NUM_EPOCHS)
        plt.plot(np.asarray(x), np.asarray(self.loss_list))
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.title('Error vs Epochs Plot')
        plt.savefig(config.OUT_DIR + 'Error_Plot.jpg')
        plt.show()
        