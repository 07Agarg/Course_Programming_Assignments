# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 05:02:08 2019

@author: Ashima
"""
import config
import data
import model
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    data = data.Data()
    #Read Train Data and load Train data
    data.read(config.INPUT_FILE_PATH)
    #visualize dataset using pairplot
    #data.visualize()
    #build model 
    model = model.Model()
    #Train using SVM CLassifier
#    model.linearsvc_train(data)
#    print("SVM trained model")
    #Train using Gaussian Naive Bayes Classifier
#    model.naive_bayes_train(data)
#    print("Naive Bayes trained model")
#    #Train using Decision trees Classifier
    model.decision_trees_train(data)
    print("Decision Trees trained model")
    
#    #Read Test File
#    data.read(config.TEST_FILE)
#    #Test using SVC
#    model.svc_test(data)
#    #Test using Naive Bayes
#    model.naive_bayes_test(data)
#    #Test using Decision Trees Classifier
#    model.decision_trees_test(data)

#    train = train_data.decode().split()
#    for i in range(np.shape(train)[0]):
#        train[i] = train[i].split(',')
#    train = pd.DataFrame.from_records(train)
#    column_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean']
#    data_means = train.iloc[:, 2:12]
#    data_means.columns = column_names
#    print(data_means.shape)
#    Y = train.iloc[:,1]
#    train.drop(train.columns[[0, 1]], axis = 1, inplace = True)
#    
#    labelEncoder_Y = LabelEncoder()
#    Y_encoded = labelEncoder_Y.fit_transform(Y)
#    print(Y.value_counts())
#    sns.pairplot(data = data_means)
    
        #train[i] = list(map(np.float64, train[i]))
    
    #Y = train[0]
    #print(type(train_data))
#    data.read(config.LABELS_FILE_PATH)
#    print("data read successfully")
#    
