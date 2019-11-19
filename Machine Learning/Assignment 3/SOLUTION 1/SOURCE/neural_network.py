# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 04:36:22 2019

@author: Ashima
"""

import os
import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.neural_network import MLPClassifier

class Network:
    def __init__(self, n_examples, h1, out):
        self.weights = []
        self.bias = []
        self.layers = [h1, 100, 50, 50, out]
        self.Z = []
        self.A = []
        self.Z_val = []
        self.A_val = []
        self.dW = []
        self.dB = []
        self.loss_lists_train = []
        self.loss_lists_val = []
        self.accuracy_list_train = []
        self.accuracy_list_val = []
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
        self.Z = []
        self.A = []
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

    def forward_val(self, X):
        self.Z_val = []
        self.A_val = []
        self.A_val.append(X)
        for i in range(len(self.layers) - 2):
            _Z = np.dot(self.weights[i], self.A_val[i]) + self.bias[i]
            self.Z_val.append(_Z)
            self.A_val.append(self.sigmoid(_Z))
        #Last layer
        _Z = np.dot(self.weights[-1], self.A_val[-1]) + self.bias[-1]
        self.Z_val.append(_Z)
        _A = self.softmax(_Z)
        self.A_val.append(_A)
    
    def backward_pass(self, Y):
        self.__reset()
        dz4 = self.A[-1] - Y
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
        
    def train(self, X, Y):
        X_train, Y_train = X[:config.NUM_TRAIN], Y[:config.NUM_TRAIN]
        X_val, Y_val = X[config.NUM_TRAIN:], Y[config.NUM_TRAIN:]
        
        for i in range(10):
            print(Y_train[i])
            
        X_train, Y_train = X_train.T, Y_train.T
        X_val, Y_val = X_val.T, Y_val.T
        
        print(Y_train.shape)
        print("n classes: ", config.CLASSES)
        Y_train_encoded = np.eye(config.CLASSES)[Y_train.astype('int32')].T.reshape(config.CLASSES, Y_train.shape[1])
        Y_val_encoded = np.eye(config.CLASSES)[Y_val.astype('int32')].T.reshape(config.CLASSES, Y_val.shape[1])
        
        print(Y_train_encoded.shape)

        #Initialize parameters
        self.initialize_parameters()
        
        #Start training
        for epoch in range(config.NUM_EPOCHS):
            self.forward_pass(X_train)
            train_loss = self.compute_loss(Y_train_encoded, self.A[-1])
            self.forward_val(X_val)
            val_loss = self.compute_loss(Y_val_encoded, self.A_val[-1])
            print("Epoch: {}, Train loss: {} ".format(epoch, train_loss))
            print("Epoch: {}, Validation loss: {} ".format(epoch, val_loss))
            self.backward_pass(Y_train_encoded)
            self.update_weights()
            self.loss_lists_train.append(train_loss)
            self.loss_lists_val.append(val_loss)            
            self.calculate_accuracy(X_train, Y_train, "None", True)    
            self.calculate_accuracy(X_val, Y_val, "None", False)
            
        #Calculate overall accuracy and loss
        self.calculate_accuracy(X_train, Y_train, "Train", True)
        print("Final Train Loss: {}".format(train_loss))
        self.calculate_accuracy(X_val, Y_val, "Validation", False)
        print("Final Validation Loss: {}".format(val_loss))
        
        self.save_weights()
    
    def test(self, X, Y):
        Y_encoded = np.eye(config.CLASSES)[Y.astype('int32')].T.reshape(config.CLASSES, Y.shape[1])
        self.predict(X)
        loss = self.compute_loss(Y_encoded, self.A[-1])
        print("Final Test Loss: {} ".format(loss))
        self.calculate_accuracy(X, Y, "Test", False)
    
    def predict(self, X):
        self.forward_pass(X)
        Y_pred = np.argmax(self.A[-1].T, axis = 1)
        return Y_pred
        
    def calculate_accuracy(self, X, Y, string, boolval):
        accuracy = 0
        Y = Y.T
        Y_pred = self.predict(X) 
        for i in range(X.shape[1]):
            accuracy += (Y_pred[i] == Y[i])
        if string == "None":
            if boolval == True:
                self.accuracy_list_train.append((accuracy/Y.shape[0])*100)
            else:
                self.accuracy_list_val.append((accuracy/Y.shape[0])*100)
        else:
            print("{} Set Accuracy {}".format(string, (accuracy/Y.shape[0])*100))
    
    def save_weights(self):
        print(np.shape(self.weights))
        np.save(os.path.join(config.OUT_DIR, config.WEIGHTS_FILE), self.weights)
    
    def sklearn_train(self, X, Y):
        mlp = MLPClassifier(hidden_layer_sizes = (100, 50, 50), activation = 'logistic', solver = 'sgd', max_iter = config.NUM_EPOCHS, learning_rate_init = config.LEARNING_RATE)
        classifier = mlp.fit(X, Y)
        #Y_predict_proba = classifier.predict_proba(X)
        train_accuracy = classifier.score(X, Y)
        print("Accuracy Score(Using Sklearn): {} ".format(train_accuracy))
        train_loss = classifier.loss_
        print("Loss(Using Sklearn): {} ".format(train_loss))
    
    def plot_cost(self, string):
        x = np.arange(config.NUM_EPOCHS)
        if string == 'Train':
            plt.plot(np.asarray(x), np.asarray(self.loss_lists_train), color = 'red')
        else:
            plt.plot(np.asarray(x), np.asarray(self.loss_lists_val), color = 'red')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Loss vs Iterations Plot(" + string + ")")
        #plt.savefig(config.OUT_DIR + 'Loss_Curve_' + string + '.jpg')
        plt.show()
        
    def plot_accuracy(self, string):
        x = np.arange(config.NUM_EPOCHS)
        if string == 'Train':
            plt.plot(np.asarray(x), np.asarray(self.accuracy_list_train), color = 'blue')
        else:
            plt.plot(np.asarray(x), np.asarray(self.accuracy_list_val), color = 'blue')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.title("Accuracy vs Iterations Plot(" + string + ")")
        #plt.savefig(config.OUT_DIR + 'Accuracy_Plot_' + string + '.jpg')
        plt.show()
        
    def plot_tsne(self):
        file = os.path.join(config.OUT_DIR, config.WEIGHTS_FILE)
        print(file)
        weights = np.load(file + ".npy", allow_pickle = True)
        print(type(weights))
        tsne = manifold.TSNE(n_components = 2)        
        print("weights shape: ", weights.shape)
        transformed_weights = tsne.fit_transform(weights[-1])
        plt.scatter(transformed_weights[0], transformed_weights[1], cmap = plt.cm.rainbow)
        plt.title('T-SNE Plot')
        #plt.savefig(config.OUT_DIR + 'T-SNE Plot')
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