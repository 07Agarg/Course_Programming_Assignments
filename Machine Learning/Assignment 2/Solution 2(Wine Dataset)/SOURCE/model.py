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
from sklearn.tree import DecisionTreeClassifier
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
        self.cm_train = None
        self.cm_test = None
    
    def preprocess_data(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
    
    def class_accuracy(self, cmatrix, string):
        #print("Accuracy of each class in {} Set".format(string))
        cmatrix_sum = np.sum(cmatrix, axis = 1)
        for i in range(config.CLASSES):
            acc = cmatrix[i][i]/cmatrix_sum[i]
            print("Accuracy of class {} {}".format(i, acc))
            
    def visualize_data(self):
        dataset = pd.DataFrame(np.c_[self.X, self.Y], columns = np.append(self.feature_names, ['target']))
        #print(dataset)
        sns.pairplot(data = dataset, vars = dataset.columns[:-1], hue = 'target')
        #print("pairplot done")
    
    def linearsvc_ovo(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.3, random_state = 42)
        start_time = time.time()
        #self.svc_ovo_clf = SVC(kernel = 'linear',decision_function_shape = 'ovo', probability = True)
        self.svc_ovo_clf = SVC(kernel = 'linear', probability = True)  #Default C = 1.0
        self.svc_ovo_clf.fit(X_train, Y_train)
        train_accuracy = self.svc_ovo_clf.score(X_train, Y_train)
        end_time = time.time()
        
        self.cm_train = confusion_matrix(Y_train, self.svc_ovo_clf.predict(X_train))
        self.class_accuracy(self.cm_train, "Linear SVC(One Vs One)-Train")
        Y_predict = self.svc_ovo_clf.predict(X_test)
        self.cm_test = confusion_matrix(Y_test, self.svc_ovo_clf.predict(X_test))
        self.class_accuracy(self.cm_test, "Linear SVC(One Vs One)-Test")
        
        print("Train Set Accuracy Score (Linear SVM-One Vs One): {} ".format(train_accuracy))
        print("Test Set Accuracy Score (Linear SVM-One Vs One): {} ".format(self.svc_ovo_clf.score(X_test, Y_test)))
        print("F1 Score (Linear SVM): {} ".format(f1_score(Y_test, Y_predict, average = 'micro')))
        print("Execution Time (Linear SVM-One Vs One): {0:.5} seconds ".format(end_time - start_time))
        print("Plot ROC Curve for (Linear SVM Classifier)")
        self.plot_roc("Linear SVM-One Vs One", self.svc_ovo_clf)
        print("\n\n")
    
    def linearsvc_ovr(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.3, random_state = 42)
        start_time = time.time()
        #self.svc_ovr_clf = SVC(kernel = 'linear', decision_function_shape = 'ovr', probability = True)
        self.svc_ovr_clf = LinearSVC()   #,probability = True)  #Default C = 1.0, multi_class = ovr(one-vs-rest)
        self.svc_ovr_clf.fit(X_train, Y_train)
        train_accuracy = self.svc_ovr_clf.score(X_train, Y_train)
        Y_predict = self.svc_ovr_clf.predict(X_test)
        end_time = time.time()
        
        self.cm_train = confusion_matrix(Y_train, self.svc_ovr_clf.predict(X_train))
        self.class_accuracy(self.cm_train, "Linear SVC(One Vs Rest)-Train")
        self.cm_test = confusion_matrix(Y_test, self.svc_ovr_clf.predict(X_test))
        self.class_accuracy(self.cm_test, "Linear SVC(One Vs Rest)-Test")
        
        print("Train Set Accuracy Score (Linear SVM-One Vs Rest): {} ".format(train_accuracy))
        print("Train Set Accuracy Score (Linear SVM-One Vs Rest): {} ".format(train_accuracy))
        print("Test Set Accuracy Score (Linear SVM-One Vs Rest): {} ".format(self.svc_ovr_clf.score(X_test, Y_test)))
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
        
        self.cm_train = confusion_matrix(Y_train, self.naivebayes_clf.predict(X_train))
        self.class_accuracy(self.cm_train, "Gaussian naive Bayes-Train")
        self.cm_test = confusion_matrix(Y_test, self.naivebayes_clf.predict(X_test))
        self.class_accuracy(self.cm_test, "Gaussian naive Bayes-Test")
        
        print("Train Set Accuracy Score (Naive Bayes): {} ".format(train_accuracy))
        print("Test Set Accuracy Score (Naive Bayes): {} ".format(self.naivebayes_clf.score(X_test, Y_test)))
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
        
        self.cm_train = confusion_matrix(Y_train, self.dec_tree_clf.predict(X_train))
        self.class_accuracy(self.cm_train, "Decision Tree-Train")
        self.cm_test = confusion_matrix(Y_test, self.dec_tree_clf.predict(X_test))
        self.class_accuracy(self.cm_test, "Decision Tree-Test")
        
        print("Train Set Accuracy Score (Decision tree classifier): {} ".format(train_accuracy))
        print("Test Set Accuracy Score (Decision tree classifier): {} ".format(self.dec_tree_clf.score(X_test, Y_test)))
        print("F1 Score (Decision tree classifier): {} ".format(f1_score(Y_test, Y_predict, average = 'micro')))
        print("Execution Time (Decision tree classifier): {0:.8f} seconds ".format(end_time - start_time))
        print("Plot ROC Curve for (Decision tree classifier)")
        self.plot_roc("Decision Tree-Depth- " + str(config.DECISION_TREE_DEPTH), self.dec_tree_clf)
        print("\n\n")
        
    def plot_roc(self, string, clf):
        Y_test = label_binarize(self.Y, classes = config.CLASS_LABELS)
        #print(Y_test)
        if clf == self.svc_ovr_clf:
            Y_predict = clf.decision_function(self.X)
        else:
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