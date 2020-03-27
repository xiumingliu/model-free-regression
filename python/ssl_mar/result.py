# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:49:52 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

ratio_average_mar = np.load('ratio_average_mar.npy')
ratio_average_mcar = np.load('ratio_average_mcar.npy')

bins = np.arange(0, 1, 0.05)

plt.figure(figsize=(5, 5))
plt.plot(bins[1:], ratio_average_mar, 'b-', label = r'\textsc{Mar}')
plt.plot(bins[1:], ratio_average_mcar, 'r:', label = r'\textsc{Mcar}')
plt.plot(bins[1:], bins[1:], '--', color='gray')
plt.grid()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(np.arange(0, 1.2, step=0.2))
plt.yticks(np.arange(0, 1.2, step=0.2))
plt.xlabel(r"$p$")
plt.ylabel(r"E$[\ \pi(\bm{x})\ |\ \hat{\pi}(\bm{x}) = p\ ]$")
plt.legend()
plt.tight_layout()
plt.savefig("reliability_diagram_mnist.png")
plt.savefig("reliability_diagram_mnist.pdf", format='pdf')