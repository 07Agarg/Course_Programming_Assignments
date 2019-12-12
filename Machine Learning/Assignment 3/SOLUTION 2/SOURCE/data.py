# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 03:36:55 2019

@author: Ashima
"""

import os
import pickle
import config
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
#import torch.utils.data as utils

class Data:
    def __init__(self):
        self.dataloader = None
        self.size = 0
        self.dataX_features = []
        self.dataY = []
        
    def preprocess(self, data_size, dataX, dataY):
        dataX = np.reshape(dataX, (data_size, config.IMAGE_SIZE, config.IMAGE_SIZE, config.CHANNELS))
        dataY = np.reshape(dataY, (dataY.shape[0], 1))
        print("dataX reshaped shape: ", dataX.shape)
        print("dataY reshaped shape: ", dataY.shape)
        dataX = np.array(dataX)
        data_transforms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    
        dataX_transformed = torch.stack([data_transforms(i) for i in dataX])
        print("dataX_transformed shape: ", dataX_transformed.shape)
        
        tensor_x = torch.stack([i for i in dataX_transformed])          
        tensor_y = torch.stack([torch.Tensor(i) for i in dataY])
        
        dataset = TensorDataset(tensor_x, tensor_y)                     
        dataloader = DataLoader(dataset)                               
        self.dataloader = dataloader
        return dataloader
    
    def extract_features(self, net, filename):
        count = 0
        for data in self.dataloader:
            X, Y = data
            print(Y)
            Y1 = net.forward(X).tolist()[0]
            self.dataX_features.append(Y1)
            self.dataY.append(Y.tolist()[0][0])
            count += 1
            if count % 300 == 0:
                print("Extracted {} features".format(count))

        print(np.shape(self.dataX_features))
        print(np.shape(self.dataY))
        data = {'X': self.dataX_features, 'Y': self.dataY}
        #with open(os.path.join(config.DATA_DIR, filename) , "wb") as file:
        #    pickle.dump(data, file)
            
    def read(self, filename):
        file = os.path.join(config.DATA_DIR, filename)
        with open(file, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            dataX = dict['X']
            dataY = dict['Y']
            dataX = np.asarray(dataX)
            dataY = np.asarray(dataY)
        #return dict
        dataloader = self.preprocess(dataX.shape[0], dataX, dataY)
        return dataloader
      
    def read_features(self, filename, train = True):
        file = os.path.join(config.DATA_DIR, filename)
        with open(file, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            dataX = dict['X']
            dataY = dict['Y']
            dataX = np.asarray(dataX)
            dataY = np.asarray(dataY)
        self.dataX_features = dataX
        self.dataY = dataY
        return self.dataY
            
    def get_data(self):
        return self.dataX_features, self.dataY
    