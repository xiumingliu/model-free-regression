# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:39:17 2019

@author: Administrator
"""

import scipy.io as sio
import numpy as np

data = sio.loadmat('data_new.mat')

sample = []
for i in range(1, 32, 1):
    if i < 10:
        sample.append( data.get('S0' + str(i)) )
    else:
        sample.append( data.get('S' + str(i)) )

sample = np.array(sample)


X = sample[:, :, np.nonzero(np.arange(23) != 16)[0]]
y = sample[:, :, np.nonzero(np.arange(23) == 16)[0]]
