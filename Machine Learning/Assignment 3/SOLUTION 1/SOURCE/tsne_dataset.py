# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 03:26:15 2019

@author: Ashima
"""

import os
import h5py
import config
import numpy as np
import pandas as pd
import time as time
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

filename = os.path.join(config.DATA_DIR, 'MNIST_Subset.h5')

data = h5py.File(filename, 'r+') 
print(np.shape(data))
X = data['X'][:]
y = data['Y'][:]

X = X.reshape(X.shape[0], -1)
print(X.shape)
print(y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

data_subset = df[feat_cols].values

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.3
)


#References: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b