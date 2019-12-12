# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 03:36:55 2019

@author: Ashima
"""

import config
import time
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix

class Model:
    
    def __init__(self):
        self.classifier = None
    
    def class_accuracy(self, cmatrix, string):
        print("Accuracy of each class in {} Set".format(string))
        cmatrix_sum = np.sum(cmatrix, axis = 1)
        for i in range(config.CLASSES):
            acc = cmatrix[i][i]/cmatrix_sum[i]
            print("Accuracy of class {} {}".format(i, acc))
    
    def svc_train(self, data): 
        X_train, Y_train = data.get_data()
        print("No of examples to train: ", X_train.shape[0])
        start_time = time.time()
        self.classifier = SVC(kernel = 'linear', probability  = True)
        self.classifier.fit(X_train, Y_train)
        end_time = time.time()  
        print("Training time: {}".format(end_time - start_time))
        print("Train Accuracy: ", self.classifier.score(X_train, Y_train))
        y_predict = self.classifier.predict(X_train)
        y_predict_proba = self.classifier.predict_proba(X_train)
        print("y_predict_proba train: ", y_predict_proba.shape)
        print("y_predict train: " , y_predict.shape)
        print("Confusion Matrix of Train Set")
        cm = confusion_matrix(Y_train, y_predict)
        print(cm)
        self.plot_roc(Y_train, y_predict_proba[:,1], "Train")
            
    def svc_test(self, data):
        X_test, Y_test = data.get_data()
        print("Test accuracy: ", self.classifier.score(X_test, Y_test))
        y_predict = self.classifier.predict(X_test)
        y_predict_proba = self.classifier.predict_proba(X_test)
        print("Confusion Matrix of Test Set")
        cm = confusion_matrix(Y_test, y_predict)
        print(cm)
        #y_test_binarize = label_binarize(Y_test, classes = list(config.CLASS_LABELS.keys()))
        self.plot_roc(Y_test, y_predict_proba[:, 1], "Test")
        
    def plot_roc(self, y_test, y_score, string):
        print("y_test shape: ", y_test.shape)
        print("y score shape: ", y_score.shape)
        print("ROC Curve")
        fpr = dict()
        tpr = dict()
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label = 'ROC Curve Class - ')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Plot for' + str(string))
        plt.legend()
        plt.savefig(config.OUT_DIR + 'ROC_PLOT_' + str(string) + '.jpg')
        plt.show()    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------      
# class FeatureExtractor(nn.Module):
#     def __init__(self, cnn):
#         super(FeatureExtractor, self).__init__()
#         self.features = nn.Sequential(*list(cnn.features.children()), 
#                                   *list(cnn.avgpool.children()), *list(cnn.classifier.children()))
#         
#     def forward(self, x):
#         return self.features(x)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------      
