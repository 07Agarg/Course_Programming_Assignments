# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 03:26:15 2019

@author: Ashima
"""
import os
import h5py
import config
import numpy as np

filename = os.path.join(config.DATA_DIR, 'MNIST_Subset.h5')
# =============================================================================
# 
# with h5py.File(filename, 'r') as f:
#     # List all groups
#     print("Keys: %s" % f.keys())
#     a_group_key = list(f.keys())[0]
# 
#     # Get the data
#     data = list(f[a_group_key])
# =============================================================================
    
data = h5py.File(filename, 'r+') 
print(np.shape(data))
X = data['X'][:]
Y = data['Y'][:]