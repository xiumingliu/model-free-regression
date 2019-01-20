# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:34:50 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification

cm = plt.cm.PiYG
cm_bright = ListedColormap(['#b30065', '#178000'])

data = make_moons(n_samples=1000, noise=0.1, random_state=0)
plt.figure(figsize=(5, 5))
plt.scatter(data[0][:, 0], data[0][:, 1], c = data[1], cmap='jet')

