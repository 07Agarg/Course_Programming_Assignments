# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:25:17 2019

@author: ashima
"""

import numpy as np

###GIVEN MATRICES
U = np.array([[10, 15, 1], [8, 3, 1], [11, 17, 1], [5, 11, 1], [6, 13, 1]])
X = np.array([[33, 20, 1], [18, 7, 1], [37, 22, 1], [20, 13, 1], [23, 16, 1]])

#U = np.transpose(U)
#X = np.transpose(X)

U_trans = np.transpose(U)

### PSEUDO INVERSE 
# X = UT
#U_trans*X = (U_trans*U)*T
#T = inv(U_trans*U) * (U_trans*X)

Y1 = np.dot(U_trans, X)          #Y1 = (U_trans*X)
Y2 = np.dot(U_trans, U)          #Y2 = (U_trans*U)

T = np.dot(np.linalg.inv(Y2) , Y1)
c = np.savetxt('TransformationMatrix1.txt', T)
