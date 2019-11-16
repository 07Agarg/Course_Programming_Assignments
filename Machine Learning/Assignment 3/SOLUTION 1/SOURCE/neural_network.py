# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 04:36:22 2019

@author: Ashima
"""

import config
import numpy as np

class Network:
    def __init__(self, n_examples, h1):
        self.weights = []
        self.bias = []
        self.layers = [h1, 100, 50, 50, 10]
        self.Z = []
        self.A = []
        self.dZ = []
        self.dA = []
        self.dW = []
        self.dB = []
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
            #print("shape of Z: ", _Z.shape)
            self.Z.append(_Z)
            _A = self.sigmoid(_Z)
            #print("shape of A: ", _A.shape)
            self.A.append(_A)
        #Last layer
        #print("Last layer")
        _Z = np.dot(self.weights[-1], self.A[-1]) + self.bias[-1]
        self.Z.append(_Z)
        #print("shape of Z: ", _Z.shape)
        _A = self.softmax(_Z)
        self.A.append(_A)
        #print("shape of A: ", _A.shape)
    
    def backward_pass(self, Y):
        self.__reset()
        dz4 = self.A[-1] - Y
        self.dZ.append(dz4)
        dw4 = (1./self.n_examples)*(np.matmul(dz4, self.A[-2].T))
        db4 = (1./self.n_examples)*(np.sum(dz4, axis = 1, keepdims = True))
        self.dW.append(dw4)
        self.dB.append(db4)
        
#        print("dz4 shape: ", dz4.shape)
#        print("dw4 shape: ", dw4.shape)
#        print("db4 shape: ", db4.shape)
        
        da3 = np.matmul(self.weights[-1].T, dz4)
        dz3 = da3*self.sigmoid_derivative(self.Z[-2])
        dw3 = (1./self.n_examples)*(np.matmul(dz3, self.A[-3].T))
        db3 = (1./self.n_examples)*(np.sum(dz3, axis = 1, keepdims = True))
        self.dW.insert(0, dw3)
        self.dB.insert(0, db3)
        
#        print("da3 shape: ", da3.shape)
#        print("dz3 shape: ", dz3.shape)
#        print("dw3 shape: ", dw3.shape)
#        print("db3 shape: ", db3.shape)
        
        da2 = np.matmul(self.weights[-2].T, dz3)
        dz2 = da2*self.sigmoid_derivative(self.Z[-3])
        dw2 = (1./self.n_examples)*(np.matmul(dz2, self.A[-4].T))
        db2 = (1./self.n_examples)*(np.sum(dz2, axis = 1, keepdims = True))
        self.dW.insert(0, dw2)
        self.dB.insert(0, db2)
        
#        print("da2 shape: ", da2.shape)        
#        print("dz2 shape: ", dz2.shape)
#        print("dw2 shape: ", dw2.shape)
#        print("db2 shape: ", db2.shape)
        
        da1 = np.matmul(self.weights[-3].T, dz3)
        dz1 = da1*self.sigmoid_derivative(self.Z[-4])
        dw1 = (1./self.n_examples)*(np.matmul(dz1, self.A[-5].T))
        db1 = (1./self.n_examples)*(np.sum(dz1, axis = 1, keepdims = True))
        self.dW.insert(0, dw1)
        self.dB.insert(0, db1)
        
#        print("da1 shape: ", da1.shape) 
#        print("dz1 shape: ", dz1.shape)
#        print("dw1 shape: ", dw1.shape)
#        print("db1 shape: ", db1.shape)
        
    def update_weights(self):
        l = len(self.layers)
        for i in range(l-1):
            self.weights[i] = self.weights[i] - config.LEARNING_RATE * (self.dW[i])
            self.bias[i] = self.bias[i] - config.LEARNING_RATE * (self.dB[i])
        
    def model(self, X, Y):
        Y_encoded = np.eye(config.CLASSES)[Y.astype('int32')].T.reshape(config.CLASSES, Y.shape[1])
        self.initialize_parameters()
        total_loss = 0
        for epoch in range(config.NUM_EPOCHS):
            self.forward_pass(X)
            loss = self.compute_loss(Y_encoded, self.A[-1])
            print("Epoch: {}, loss: {} ".format(epoch, loss))
            self.backward_pass(Y_encoded)
            self.update_weights()
            total_loss += loss
        