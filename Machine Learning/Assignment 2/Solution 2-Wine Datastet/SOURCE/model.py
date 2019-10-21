# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 05:02:34 2019

@author: Ashima
"""

import config
import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize

class Model():
    
    def __init__(self, data):
        self.X = data.data
        self.Y = data.target
        self.feature_names = data.feature_names
        self.svc_ovo_clf = None
        self.svc_ovr_clf = None
        self.naivebayes_clf = None
        self.dec_tree_clf = None
    
    def preprocess_data(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
    def visualize_data(self):
        dataset = pd.DataFrame.from_records(self.X)
        dataset.columns = self.feature_names
        print("Feature names: ", dataset.columns)
        print(dataset.shape)
        #sns.pairplot(data = dataset)
        #print("pairplot done")
    
    def linearsvc_ovo(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.3, random_state = 42)
        start_time = time.time()
        self.svc_ovo_clf = OneVsOneClassifier(LinearSVC())  #Default C = 1.0
        self.svc_ovo_clf.fit(X_train, Y_train)
        train_accuracy = self.svc_ovo_clf.score(X_train, Y_train)
        Y_predict = self.svc_ovo_clf.predict(X_test)
        end_time = time.time()
        print("Train Set Accuracy Score (Linear SVM-One Vs One): {} ".format(train_accuracy))
        print("Y predict shape: ", Y_predict.shape)
        print("Y_test shape: ", Y_test.shape)
        print("Test Set Accuracy Score (Linear SVM-One Vs One): {} ".format(accuracy_score(Y_test, Y_predict)))
        print("F1 Score (Linear SVM): {} ".format(f1_score(Y_test, Y_predict, average = 'micro')))
        print("Execution Time (Linear SVM-One Vs One): {0:.5} seconds ".format(end_time - start_time))
        print("Plot ROC Curve for (Linear SVM Classifier)")
        self.plot_roc("Linear SVM-One Vs One", self.svc_ovo_clf)
        print("\n\n")
    
    def linearsvc_ovr(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.3, random_state = 42)
        start_time = time.time()
        self.svc_ovr_clf = OneVsRestClassifier(SVC(probability = True))  #Default C = 1.0, multi_class = ovr(one-vs-rest)
        self.svc_ovr_clf.fit(X_train, Y_train)
        train_accuracy = self.svc_ovr_clf.score(X_train, Y_train)
        Y_predict = self.svc_ovr_clf.predict(X_test)
        end_time = time.time()
        print("Train Set Accuracy Score (Linear SVM-One Vs Rest): {} ".format(train_accuracy))
        print("Test Set Accuracy Score (Linear SVM-One Vs Rest): {} ".format(accuracy_score(Y_test, Y_predict)))
        print("F1 Score (Linear SVM-One Vs Rest): {} ".format(f1_score(Y_test, Y_predict, average = 'micro')))
        print("Execution Time (Linear SVM-One Vs Rest): {0:.5} seconds ".format(end_time - start_time))
        print("Plot ROC Curve for (Linear SVM Classifier)")
        self.plot_roc("Linear SVM-One Vs Rest", self.svc_ovr_clf)
        print("\n\n")
    
    def naive_bayes(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.3, random_state = 42)
        start_time = time.time()
        self.naivebayes_clf = GaussianNB()
        self.naivebayes_clf.fit(X_train, Y_train)
        train_accuracy = self.naivebayes_clf.score(X_train, Y_train)
        Y_predict = self.naivebayes_clf.predict(X_test)
        end_time = time.time()
        print("Train Set Accuracy Score (Naive Bayes): {} ".format(train_accuracy))
        print("Test Set Accuracy Score (Naive Bayes): {} ".format(accuracy_score(Y_test, Y_predict)))
        print("F1 Score (Naive Bayes): {} ".format(f1_score(Y_test, Y_predict, average = 'micro')))
        print("Execution Time (Naive Bayes): {0:.5} seconds ".format(end_time - start_time))
        print("Plot ROC Curve for (Naive Bayes Classifier)")
        self.plot_roc("Naive Bayes", self.naivebayes_clf)
        print("\n\n")
        
    def decision_trees(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.3, random_state = 42)
        start_time = time.time()
        self.dec_tree_clf = DecisionTreeClassifier(max_depth = config.DECISION_TREE_DEPTH)
        self.dec_tree_clf.fit(X_train, Y_train)
        train_accuracy = self.dec_tree_clf.score(X_train, Y_train)
        Y_predict = self.dec_tree_clf.predict(X_test)
        end_time = time.time()
        print("Train Set Accuracy Score (Decision tree classifier): {} ".format(train_accuracy))
        print("Test Set Accuracy Score (Decision tree classifier): {} ".format(accuracy_score(Y_test, Y_predict)))
        print("F1 Score (Decision tree classifier): {} ".format(f1_score(Y_test, Y_predict, average = 'micro')))
        print("Execution Time (Decision tree classifier): {0:.5} seconds ".format(end_time - start_time))
        print("Plot ROC Curve for (Decision tree classifier)")
        self.plot_roc("Decision Tree-Depth- " + str(config.DECISION_TREE_DEPTH), self.dec_tree_clf)
        print("\n\n")
        
    def plot_roc(self, string, clf):
        Y_test = label_binarize(self.Y, classes = config.CLASS_LABELS)
        Y_predict = clf.predict_proba(self.X)
        fpr = dict()
        tpr= dict()
        for i in range(config.CLASSES):
            fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_predict[:, i])
        plt.figure()
        colors = ['r', 'b', 'g']
        for i in range(config.CLASSES):
            plt.plot(fpr[i], tpr[i], label = 'ROC Curve Class - ' + str(i), color = colors[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Plot '+string)
        plt.legend(loc='lower right')
        plt.savefig(config.OUT_DIR + 'ROC_PLOT '+string+".jpg")
        plt.show()