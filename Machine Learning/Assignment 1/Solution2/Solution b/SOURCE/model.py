# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 05:52:25 2019

@author: ashima
"""
import config
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier

class Model():
    
    def __init__(self, size):
        self.size = size
        self.classifier = None
        self.cm = None
        self.y_predict = None
    
    def class_accuracy(self, cmatrix, string):
        print("Accuracy of each class in {} Set".format(string))
        cmatrix_sum = np.sum(cmatrix, axis = 1)
        for i in range(config.CLASSES):
            acc = cmatrix[i][i]/cmatrix_sum[i]
            print("Accuracy of class {} {}".format(i, acc))
    
    def Logistic_sgdclassifier_L1train(self, data):
        X_train, y_train = data.get_train()
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.ravel()
        sgd = SGDClassifier(loss = 'log', penalty = 'l1', max_iter = 10)
        self.classifier = sgd.fit(X_train, y_train)
        y_predict = self.classifier.predict(X_train)
        y1 = self.classifier.predict_proba(X_train)
        for i in range(10):
            print("y1 ", y1[i])
        cm = confusion_matrix(y_train, y_predict)
        self.cm = cm
        print("Confusion Matrix For Train set")
        print(cm)
        self.class_accuracy(cm, "Train")
        train_accuracy = self.classifier.score(X_train, y_train)
        print("Train Set Accuracy Score-L1 Regularized: {} ".format(train_accuracy))
        
    
    def LogisticReg_L1_train(self, data, solver):
        print("L1 Logistic Regression")
        X_train, y_train = data.get_train()
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
        lr = LogisticRegression(penalty = 'l1', max_iter = 100, multi_class = 'ovr', solver = solver)   #default C = 1.0 which is inverse of regualrization strnegth.
        self.classifier = lr.fit(X_train, y_train)
        y_predict = self.classifier.predict(X_train)
        y1 = self.classifier.predict_proba(X_train)
        for i in range(10):
            print("y1 ", y1[i])
        cm = confusion_matrix(y_train, y_predict)
        self.cm = cm
        print("Confusion Matrix For Train set")
        print(cm)
        self.class_accuracy(cm, "Train")
        train_accuracy = self.classifier.score(X_train, y_train)
        print("Train Set Accuracy Score-L1 Regularized: {} ".format(train_accuracy))
        
        
    def LogisticReg_L1_test(self, data):
        X_test, y_test = data.get_test()
        y_predict = self.classifier.predict(X_test)
        self.y_predict = y_predict
        cm = confusion_matrix(y_test, y_predict)
        print("Confusion Matrix For test set")
        #print(cm)
        self.cm = cm
        self.class_accuracy(cm, "Test")
        print("Test Set Accuracy Score-L1 Regularized: {} ".format(accuracy_score(y_test, y_predict)))
        
    def LogisticReg_L2_train(self, data, solver):
        print("L2 Logistic Regression")
        X_train, y_train = data.get_train()
        lr = LogisticRegression(penalty = 'l2', max_iter = 100, multi_class = 'ovr', solver = solver, tol=0.1)   #default C = 1.0 which is inverse of regualrization strnegth. 
        self.classifier = lr.fit(X_train, y_train)
        cm = confusion_matrix(y_train, self.classifier.predict(X_train))
        #print("Confusion Matrix For Train set")
        #print(cm)
        self.cm = cm
        self.class_accuracy(cm, "Train")
        train_accuracy = self.classifier.score(X_train, y_train)
        print("Train Set Accuracy Score-L2 Regularized: {} ".format(train_accuracy))
        
    def LogisticReg_L2_test(self, data):
        X_test, y_test = data.get_test()
        y_predict = self.classifier.predict_proba(X_test)
        print("y_predict shape: ", y_predict.shape)
        self.predict = y_predict
        cm = confusion_matrix(y_test, self.classifier.predict(X_test))
        print("Confusion Matrix For test set")
        print(cm)
        self.cm = cm
        self.class_accuracy(cm, "Test")
        print("Test Set Accuracy Score-L2 Regularized : {} ".format(accuracy_score(y_test, self.classifier.predict(X_test))))
        #y_score = self.classifier.decision_function(X_test)
        y_test_binarize = label_binarize(y_test, classes = config.class_labels)
        self.plot_roc(y_test_binarize, self.predict)
    
    def plot_roc(self, y_test, y_score):
        print("Plot ROC Curve for Each Class")
        fpr = dict()
        tpr= dict()
        for i in range(config.CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        plt.figure()
        for i in range(config.CLASSES):
            plt.plot(fpr[i], tpr[i], label = 'ROC Curve Class - ' + str(i))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Plot for All Classes')
        plt.legend()
        plt.savefig(config.OUT_DIR + 'ROC_PLOT.jpg')
        plt.show()
        """
        total_sum = np.sum(self.cm)
        for i in range(config.CLASSES):
            class_sum = np.sum(self.cm[i])
            TPR = self.cm[i][i]/class_sum
            FPR = (np.sum(self.cm[:, i]) - self.cm[i][i])/(total_sum - class_sum)
            plt.plot(FPR, TPR, label = "Class "+ str(i))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])        
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.title("ROC Plot for All Classes")
        plt.legend()
        plt.savefig(config.OUT_DIR + 'ROC_PLOT.jpg')
        plt.show()
        """    