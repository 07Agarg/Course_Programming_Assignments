# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 05:02:34 2019

@author: Ashima
"""

import config
import numpy as np
#from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class Model():
    
    def __init__(self):
        self.svc_clf = None
        self.naivebayes_clf = None
        self.dec_tree_clf = None
    
    def linearsvc_train(self, data):
        X, Y = data.get_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
        self.svc_clf = LinearSVC()
        self.svc_clf.fit(X_train, Y_train)
        train_accuracy = self.svc_clf.score(X_train, Y_train)
        print("Train Set Accuracy Score (SVM): {} ".format(train_accuracy))
        Y_predict = self.svc_clf.predict(X_test)
        print("Test Set Accuracy Score (SVM): {} ".format(accuracy_score(Y_test, Y_predict)))
        print("F1 Score (Linear SVM): {} ".format(f1_score(Y_test, Y_predict)))
        
    def naive_bayes_train(self, data):
        X, Y = data.get_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
        self.naivebayes_clf = GaussianNB()
        self.naivebayes_clf.fit(X_train, Y_train)
        train_accuracy = self.naivebayes_clf.score(X_train, Y_train)
        print("Train Set Accuracy Score (Naive Bayes): {} ".format(train_accuracy))
        Y_predict = self.naivebayes_clf.predict(X_test)
        print("Test Set Accuracy Score (Naive Bayes): {} ".format(accuracy_score(Y_test, Y_predict)))
        print("F1 Score (Naive Bayes): {} ".format(f1_score(Y_test, Y_predict)))
        
    def decision_trees_train(self, data):
        X, Y = data.get_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
        self.dec_tree_clf = DecisionTreeClassifier()
        self.dec_tree_clf.fit(X_train, Y_train)
        train_accuracy = self.dec_tree_clf.score(X_train, Y_train)
        print("Train Set Accuracy Score (Naive Bayes): {} ".format(train_accuracy))
        Y_predict = self.dec_tree_clf.predict(X_test)
        print("Test Set Accuracy Score (Naive Bayes): {} ".format(accuracy_score(Y_test, Y_predict)))
        print("F1 Score (Decision tree classifier): {} ".format(f1_score(Y_test, Y_predict)))
        
    def plot_roc(self, y_test, y_score):
        print("Plot ROC Curve for Each Class")
        fpr = dict()
        tpr= dict()
        for i in range(config.CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        plt.figure()
        for i in range(config.CLASSES):
            plt.plot(fpr[i], tpr[i], label = 'Class - ' + str(i))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Plot for All Classes')
        plt.legend(loc='lower right')
        plt.savefig(config.OUT_DIR + 'ROC_PLOT.jpg')
        plt.show()
    
#    def LogisticReg_L1_train(self, data, solver):
#        print("L1 Logistic Regression")
#        X_train, y_train = data.get_train()
#        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
#        lr = LogisticRegression(penalty = 'l1', max_iter = 100, multi_class = 'ovr', solver = solver)   #default C = 1.0 which is inverse of regualrization strnegth.
#        self.classifier = lr.fit(X_train, y_train)
#        y_predict = self.classifier.predict(X_train)
#        y1 = self.classifier.predict_proba(X_train)
#        for i in range(10):
#            print("y1 ", y1[i])
#        cm = confusion_matrix(y_train, y_predict)
#        self.cm = cm
#        print("Confusion Matrix For Train set")
#        print(cm)
#        self.class_accuracy(cm, "Train")
#        train_accuracy = self.classifier.score(X_train, y_train)
#        print("Train Set Accuracy Score-L1 Regularized: {} ".format(train_accuracy))
#        
#        
#    def LogisticReg_L1_test(self, data):
#        X_test, y_test = data.get_test()
#        y_predict = self.classifier.predict(X_test)
#        self.y_predict = y_predict
#        cm = confusion_matrix(y_test, y_predict)
#        print("Confusion Matrix For test set")
#        #print(cm)
#        self.cm = cm
#        self.class_accuracy(cm, "Test")
#        print("Test Set Accuracy Score-L1 Regularized: {} ".format(accuracy_score(y_test, y_predict)))
#        
#    def LogisticReg_L2_train(self, data, solver):
#        print("L2 Logistic Regression")
#        X_train, y_train = data.get_train()
#        lr = LogisticRegression(penalty = 'l2', max_iter = 100, multi_class = 'ovr', solver = solver, tol=0.1)   #default C = 1.0 which is inverse of regualrization strnegth. 
#        self.classifier = lr.fit(X_train, y_train)
#        cm = confusion_matrix(y_train, self.classifier.predict(X_train))
#        #print("Confusion Matrix For Train set")
#        #print(cm)
#        self.cm = cm
#        self.class_accuracy(cm, "Train")
#        train_accuracy = self.classifier.score(X_train, y_train)
#        print("Train Set Accuracy Score-L2 Regularized: {} ".format(train_accuracy))
#        
#    def LogisticReg_L2_test(self, data):
#        X_test, y_test = data.get_test()
#        y_predict = self.classifier.predict_proba(X_test)
#        self.predict = y_predict
#        cm = confusion_matrix(y_test, self.classifier.predict(X_test))
#        print("Confusion Matrix For test set")
#        print(cm)
#        self.cm = cm
#        self.class_accuracy(cm, "Test")
#        print("Test Set Accuracy Score-L2 Regularized : {} ".format(accuracy_score(y_test, self.classifier.predict(X_test))))
#        #y_score = self.classifier.decision_function(X_test)
#        y_test_binarize = label_binarize(y_test, classes = config.class_labels)
#        self.plot_roc(y_test_binarize, self.predict)