# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 05:52:05 2019

@author: ashima
"""
import config
import gzip
import os
#import cPickle
import numpy as np
image_size = 28
num_images = 5

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(a*b)
print(np.multiply(a, b))
print(np.dot(a, b))
print(np.matmul(a, b))
print(np.sum(np.multiply(a, b)))

"""
a = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
b = np.sum(a)

TPR = a[0][0]/np.sum(a[0])
print(TPR)
FPR = (np.sum(a[:, 0]) - a[0][0])/(np.sum(a) - np.sum(a[0]))
print("Den ", np.sum(a) - np.sum(a[0]))
print(FPR)
"""
#for i in range(a.shape[0]):
#    l = (a[i][i]/np.sum(axis))
#with open(os.path.join(config.DATA_DIR, 'train-images-idx3-ubyte.gz'), 'rb') as f, gzip.GzipFile(fileobj = f) as bytestream:
    #data = bytestream.read(image_size * image_size * num_images)
"""
f = gzip.open(os.path.join(config.DATA_DIR, 'train-images-idx3-ubyte.gz'),'r')
#train_data, val_data, test_data = cPickle.load(f)
#f.close()


import numpy as np
f.read(16)
buf = f.read(image_size * image_size * 60000)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(60000, image_size, image_size, 1)
#data = data.reshape(data.shape[0], -1)
data = data[:100]
print("data shape ", data.shape)
"""
"""
import matplotlib.pyplot as plt
image = np.asarray(data[2]).squeeze()
plt.imshow(image)
plt.show()

f = gzip.open(os.path.join(config.DATA_DIR,'train-labels-idx1-ubyte.gz'),'r')
f.read(8)
buf = f.read(10000)
labels = np.frombuffer(buf, dtype = np.uint8).astype(np.int64)
labels = labels.reshape(10000, 1)
"""
"""
for i in range(0,5):   
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)
"""