# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 04:36:22 2019

@author: Ashima
"""

import numpy as np

class neural_network():
    def __init__(self):
        self.weights = []
        self.bias = []
        self.layers = [784, 100, 50, 50, 10]
        self.Z = []
        self.A = []
        self.dZ = []
        self.dA = []
        self.dW = []
        self.db = []
        
    def initialize_parameters_L(self):
        l = len(self.layers)
        for i in range(1, l):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i-1]))
            self.bias.append(np.zeros(self.layers[i], 1))   
            
    def compute_loss(self, Y_true, Y_predict):
        loss = np.sum(np.multiply(Y_true, np.log(Y_predict)))
        loss = -loss/(Y_true.shape[1])
        return loss
        
    def sigmoid(self, Z):
        out = 1./(1 + np.exp(-Z))
        return out
    
    def softmax(self, Z):
        out = np.exp(Z)/np.sum(np.exp(Z), axis = 0)
        return out
        
    def forward_pass(self, X):
        self.A.append(X)
        for i in range(len(self.layers) - 1):
            Z = np.dot(self.weights[i], self.A[i]) + self.bias[i]
            self.Z.append(Z)
            self.A.append(self.sigmoid(Z))
        #Last layer
        Z = np.dot(self.weights[i], self.A[i])
        self.A.append(self.softmax(Z))
    
    def backward_pass(self, Y):
        self.dZ.append(self.A[-1] - Y)
        
        
    def model():
        pass



