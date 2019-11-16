# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 04:36:22 2019

@author: Ashima
"""

import config
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, n_examples, h1):
        self.weights = []
        self.bias = []
        self.layers = [h1, 100, 50, 50, 10]
        self.Z = []
        self.A = []
        self.dW = []
        self.dB = []
        self.loss_lists = []
        self.accuracy_list = []
        self.n_examples = n_examples
    
    def __reset(self):
        self.dW = []
        self.dB = []
        
    def initialize_parameters(self):
        l = len(self.layers)
        for i in range(1, l):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i-1]))
            self.bias.append(np.zeros((self.layers[i], 1)))   
            
    def compute_loss(self, Y_true, Y_predict):
        loss = np.sum(np.multiply(Y_true, np.log(Y_predict)))
        loss = -loss/(Y_true.shape[1])
        return loss
        
    def sigmoid(self, Z):
        out = 1./(1 + np.exp(-Z))
        return out
    
    def sigmoid_derivative(self, Z):
        out = self.sigmoid(Z)*(1 - self.sigmoid(Z))
        return out
        
    def softmax(self, Z):
        shiftZ = Z - np.max(Z)
        out = np.exp(shiftZ)/np.sum(np.exp(shiftZ), axis = 0)
        return out
    
    def softmax2(self, Z):
        out = np.exp(Z)/np.sum(np.exp(Z), axis = 0)
        return out
        
    def forward_pass(self, X):
        self.A.append(X)
        for i in range(len(self.layers) - 2):
            _Z = np.dot(self.weights[i], self.A[i]) + self.bias[i]
            self.Z.append(_Z)
            self.A.append(self.sigmoid(_Z))
        #Last layer
        _Z = np.dot(self.weights[-1], self.A[-1]) + self.bias[-1]
        self.Z.append(_Z)
        _A = self.softmax(_Z)
        self.A.append(_A)
    
    def backward_pass(self, Y):
        self.__reset()
        dz4 = self.A[-1] - Y
        self.dZ.append(dz4)
        dw4 = (1./self.n_examples)*(np.matmul(dz4, self.A[-2].T))
        db4 = (1./self.n_examples)*(np.sum(dz4, axis = 1, keepdims = True))
        self.dW.append(dw4)
        self.dB.append(db4)
        
        dz = dz4
        l = len(self.layers) - 1
        for i in range(1, l):
            da = np.matmul(self.weights[-i].T, dz)
            dz = da*self.sigmoid_derivative(self.Z[-(i+1)])
            dw = (1./self.n_examples)*(np.matmul(dz, self.A[-(i+2)].T))
            db = (1./self.n_examples)*(np.sum(dz, axis = 1, keepdims = True))
            self.dW.insert(0, dw)
            self.dB.insert(0, db)
        
    def update_weights(self):
        l = len(self.layers)
        for i in range(l-1):
            self.weights[i] = self.weights[i] - config.LEARNING_RATE * (self.dW[i])
            self.bias[i] = self.bias[i] - config.LEARNING_RATE * (self.dB[i])
        
    def train(self, X, Y, train = "True"):
        Y_encoded = np.eye(config.CLASSES)[Y.astype('int32')].T.reshape(config.CLASSES, Y.shape[1])
        self.initialize_parameters()
        for epoch in range(config.NUM_EPOCHS):
            self.forward_pass(X)
            loss = self.compute_loss(Y_encoded, self.A[-1])
            print("Epoch: {}, loss: {} ".format(epoch, loss))
            self.backward_pass(Y_encoded)
            self.update_weights()
            self.loss_lists.append(loss)
            self.calculate_accuracy(X, Y, "None")
        if train:
            print("Final Train Loss: {}".format(loss))
        else:
            print("Final Validation Loss: {}".format(loss))
    
    def test(self, X, Y):
        Y_encoded = np.eye(config.CLASSES)[Y.astype('int32')].T.reshape(config.CLASSES, Y.shape[1])
        self.forward_pass(X)
        loss = self.compute_loss(Y_encoded, self.A[-1])
        print("Final Test Loss: {} ".format(loss))
    
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
            
    def plot_cost(self, string):
        x = np.arange(config.NUM_EPOCHS)
        plt.plot(np.asarray(x), np.asarray(self.loss_lists), color = 'red')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Loss vs Iterations Plot")
        plt.savefig(config.OUT_DIR + 'Loss_Curve.jpg')
        plt.show()
        
    def plot_accuracy(self):
        x = np.arange(config.NUM_EPOCHS)
        plt.plot(np.asarray(x), np.asarray(self.accuracy_list), color = 'blue')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.title('Accuracy vs Iterations Plot')
        plt.savefig(config.OUT_DIR + 'Accuracy_Plot.jpg')
        plt.show()
        
#        da3 = np.matmul(self.weights[-1].T, dz4)
#        dz3 = da3*self.sigmoid_derivative(self.Z[-2])
#        dw3 = (1./self.n_examples)*(np.matmul(dz3, self.A[-3].T))
#        db3 = (1./self.n_examples)*(np.sum(dz3, axis = 1, keepdims = True))
#        self.dW.insert(0, dw3)
#        self.dB.insert(0, db3)
#        da2 = np.matmul(self.weights[-2].T, dz3)
#        dz2 = da2*self.sigmoid_derivative(self.Z[-3])
#        dw2 = (1./self.n_examples)*(np.matmul(dz2, self.A[-4].T))
#        db2 = (1./self.n_examples)*(np.sum(dz2, axis = 1, keepdims = True))
#        self.dW.insert(0, dw2)
#        self.dB.insert(0, db2)
#        da1 = np.matmul(self.weights[-3].T, dz3)
#        dz1 = da1*self.sigmoid_derivative(self.Z[-4])
#        dw1 = (1./self.n_examples)*(np.matmul(dz1, self.A[-5].T))
#        db1 = (1./self.n_examples)*(np.sum(dz1, axis = 1, keepdims = True))
#        self.dW.insert(0, dw1)
#        self.dB.insert(0, db1)