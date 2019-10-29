# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:02:08 2019

@author: Ashima
"""
import config
#import data
import model
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine

if __name__ == "__main__":
    #Load Data
    data = load_wine(return_X_y = False)
    #build model 
    model = model.Model(data)
    model.preprocess_data()
    #model.visualize_data()
#    #Train using SVM CLassifier_One Vs One
    model.linearsvc_ovo()
    #print("SVM trained model-One Vs One")
##    #Train using SVM CLassifier_One Vs One    
    model.linearsvc_ovr()
    #print("SVM trained model-One Vs Rest")
#   Train using Gaussian Naive Bayes Classifier
    model.naive_bayes()
    #print("Naive Bayes trained model")
    #Train using Decision trees Classifier
    model.decision_trees()
    #print("Decision Trees trained model")





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
