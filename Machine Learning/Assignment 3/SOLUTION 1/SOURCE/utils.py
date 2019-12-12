# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:44:27 2019

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

def plot_tsne():
    file = os.path.join(config.OUT_DIR, config.WEIGHTS_FILE)
    print(file)
    weights = np.load(file + ".npy", allow_pickle = True)
    print(type(weights))
    X = weights[-1].T.flatten('F')
    print("weights shape: ", X.shape)
    df1 = pd.DataFrame(X, columns = ['weights'])
    df1['y'] = ""
    
    for i in range(51):
        df1['y'][i] = 7
        
    for i in range(51, 100):
        df1['y'][i] = 9
    df1['label'] = df1['y'].apply(lambda i:str(i))
    print('Size of the dataframe: {}'.format(df1.shape))
    
    data_subset = df1['weights'].values.reshape(-1, 1)
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    df1['tsne-2d-one'] = tsne_results[:,0]
    df1['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df1,
        legend="full",
        alpha=0.3
    )