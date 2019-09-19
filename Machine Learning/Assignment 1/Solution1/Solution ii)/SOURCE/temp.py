# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:12:09 2019

@author: ashima
"""

import config
import pandas as pd
import os
import numpy as np
import random
t = np.arange(5)

test_list = np.array([1, 4, 5, 6, 3, 7, 9, 8])
print(test_list.shape)
test_list = test_list.reshape(1, test_list.shape[0])
print(test_list.shape)

random.shuffle(test_list)
print(test_list)

K = 3
extra = len(test_list)%K
remain = test_list[extra:]
x = len(test_list)-extra
new_list = test_list[:x]
#test_list  = test_list
split_lists = np.split(new_list, 3)
#split_list

for i in range(extra):
    split_lists[i] = np.append(split_lists[i], remain[i])
    #split_lists[i].append()

print(split_lists)
#print(split_lists[0])
final = []

"""
file = pd.read_csv(os.path.join(config.DATA_DIR, config.FILE_PATH) , sep = ' ', names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
print(file.shape)
print(type(file))
for i in range(2):
    print(file.iloc[i])
file = file.iloc[np.random.permutation(len(file))]

#np.random.shuffle(file)
print(file.shape)
for i in range(2):
    print(file.iloc[i])

"""

W = np.random.randn(9).reshape(9, 1)
a = ['M', 'F', 'F', 'M', 'F', 'F', 'M', 'I', 'I', 'I']
b = np.array([1, 2, 3, 4]).reshape(4, 1)
b = b + 1