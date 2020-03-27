# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:41:16 2019

@author: Xiuming
"""

import numpy as np

def data_gen(N, N_labeled, p_component, mean, cov):
    K_component = np.size(p_component)
    k = np.random.choice(K_component, p=p_component, size=N)
    X = np.zeros((N, 2))
    for n in range(N):    
        X[n, :] = np.random.multivariate_normal(mean[k[n], :], cov)
        
    Y = np.zeros(N)
    Y[k == 0] = 0
    Y[k == 1] = 1
    Y[k == 2] = np.random.randint(2, size=np.size(Y[k == 2]))
    
    index = np.arange(N)
    L = np.zeros(N)
    L[np.random.choice(index[k == 0], int(N_labeled/2), replace=False)] = 1
    L[np.random.choice(index[k == 1], int(N_labeled/2), replace=False)] = 2
    
    return X, Y, L