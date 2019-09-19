# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:10:41 2019

@author: ashima
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#2bit depth image
v = np.array([0, 1, 2, 3])
p = np.array([0.4, 0.2, 0, 0.4])

#print(np.sum([p[x] for x in range(1)]))

s = [0] * 4  #Output array

for i in range(4):
    s[i] = np.round(3*np.sum([p[x] for x in range(i+1)]))
    
    ps = [0] * 4
for (i,val) in enumerate(s):
    ps[np.int(val)] += p[i]
         
    
s = np.array(s)
fig, ax = plt.subplots(1, 2)

#Display input Histogram
ax[0].bar(v, p, width = 0.1, color = 'Red')
ax[0].set_title('Input Histogram')
ax[0].set_xlabel('r')
ax[0].set_ylabel('Pr')

#Display Output Histogram
ax[1].bar(v, ps)
ax[1].set_title('Output Histogram')
ax[1].set_xlabel('s')
ax[1].set_ylabel('Ps')
fig.savefig('Histograms')
