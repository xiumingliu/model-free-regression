# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:40:53 2019

@author: Administrator
"""

import functions

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve 

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

# =============================================================================
# Generate data
# =============================================================================
N = 1000
N_labeled = 250
p_component = np.array([.25, .25, .5])

mean = np.array([[-5, -5], [5, 5], [.0, .0]])
cov = np.array([[1, -.5], [-.5, 1]]) * 1

rv1 = multivariate_normal(mean[0, :], cov)
rv2 = multivariate_normal(mean[2, :], cov)
rv3 = multivariate_normal(mean[1, :], cov)

X, Y, L = functions.data_gen(N, N_labeled, p_component, mean, cov)

plt.figure(figsize=(5, 5))
plt.scatter(X[np.logical_and(L == 0 , Y == 0), 0], X[np.logical_and(L == 0 , Y == 0), 1], s=100, facecolors='none', marker='o', edgecolors='gray', alpha=0.5)
plt.scatter(X[np.logical_and(L == 0 , Y == 1), 0], X[np.logical_and(L == 0 , Y == 1), 1], s=100, marker='x', c='gray', alpha=0.5)
plt.scatter(X[L == 1, 0], X[L == 1, 1], marker='o', edgecolors='tab:blue', s=100, facecolors='none', label='0')
plt.scatter(X[L == 2, 0], X[L == 2, 1], marker='x', c='tab:orange', s=100, label='1')   
plt.legend() 
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig("data.png")
plt.savefig("data.pdf", format='pdf')

# =============================================================================
# Learning p(x | y=0), p(x | y=1), p(x | l=0)
# =============================================================================

p_x_y0 = BayesianGaussianMixture(n_components=1)
p_x_y0.fit(X[L==1, :])

p_x_y1 = BayesianGaussianMixture(n_components=1)
p_x_y1.fit(X[L==2, :])


p_x_l0 = BayesianGaussianMixture(n_components=3)
p_x_l0.fit(X[L==0, :])

# =============================================================================
# Predicting unlabeled samples
# =============================================================================

score_p_x_y0_l0 = p_x_y0.score_samples(X[L == 0])
score_p_x_y1_l0 = p_x_y1.score_samples(X[L == 0])
score_p_x_l0 = p_x_l0.score_samples(X[L == 0])

# =============================================================================
# LRT
# =============================================================================

R = np.zeros((np.size(L[L == 0]), 2))
R[:, 0] = score_p_x_y0_l0 - score_p_x_l0
R[:, 1] = score_p_x_y1_l0 - score_p_x_l0

# =============================================================================
# Sampling 
# =============================================================================

index_unlabel = np.where(L == 0)[0]
post = np.zeros((np.size(L[L == 0]), 2))
y_hat_x_l0 = np.zeros((np.size(L[L == 0])))
x_l0 = X[L == 0]

for i in range(np.size(index_unlabel)):
    if np.max(R[i, :]) > 0:
        post[i, 0] = np.exp(score_p_x_y0_l0[i])/(np.exp(score_p_x_y0_l0[i]) + np.exp(score_p_x_y1_l0[i]))
        post[i, 1] = np.exp(score_p_x_y1_l0[i])/(np.exp(score_p_x_y0_l0[i]) + np.exp(score_p_x_y1_l0[i]))
    else:
        post[i, 0] = 0.5
        post[i, 1] = 0.5
            
    y_hat_x_l0[i] = np.random.choice([0, 1], size=1, p=post[i, :], replace=False)
    
#error_prob = 1 - np.max(post, axis=1)
    
#plt.figure(figsize=(5, 5))
#plt.scatter(x_l0[y_hat_x_l0 == 0, 0], x_l0[y_hat_x_l0 == 0, 1], s=100, facecolors='none', marker='o', edgecolors='r', alpha=0.5)
#plt.scatter(x_l0[y_hat_x_l0 == 1, 0], x_l0[y_hat_x_l0 == 1, 1], s=100, marker='x', c='b', alpha=0.5)
#plt.scatter(X[L == 1, 0], X[L == 1, 1], marker='o', edgecolors='r', s=100, facecolors='none')
#plt.scatter(X[L == 2, 0], X[L == 2, 1], marker='x', c='b', s=100)    
#plt.xlim([-10, 10])
#plt.ylim([-10, 10])

Y_hat_mar = Y
Y_hat_mar[L == 0] = y_hat_x_l0

plt.figure(figsize=(5, 5))
plt.scatter(X[Y_hat_mar == 0, 0], X[Y_hat_mar == 0, 1], marker='o', edgecolors='r', s=100, facecolors='none')
plt.scatter(X[Y_hat_mar == 1, 0], X[Y_hat_mar == 1, 1], marker='x', c='b', s=100)    
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig("data_self_label_mar.png")
plt.savefig("data_self_label_mar.pdf", format='pdf')

# =============================================================================
# Learn the new model
# =============================================================================

p_x_y0_new = BayesianGaussianMixture(n_components=2)
p_x_y0_new.fit(X[Y_hat_mar == 0, :])

p_x_y1_new = BayesianGaussianMixture(n_components=2)
p_x_y1_new.fit(X[Y_hat_mar == 1, :])

N_test = 10000
X_test, Y_test, L_test = functions.data_gen(N_test, 0, p_component, mean, cov)

score_p_x_y0_new = p_x_y0_new.score_samples(X_test)
score_p_x_y1_new = p_x_y1_new.score_samples(X_test)



post_new = np.zeros((np.size(L_test[L_test == 0]), 2))
for i in range(N_test):
    post_new[i, 0] = np.exp(score_p_x_y0_new[i])/(np.exp(score_p_x_y0_new[i]) + np.exp(score_p_x_y1_new[i]))
    post_new[i, 1] = np.exp(score_p_x_y1_new[i])/(np.exp(score_p_x_y0_new[i]) + np.exp(score_p_x_y1_new[i]))
    
y_hat_mar_test = np.argmax(post_new, axis=1)    
#plt.figure()
#plt.scatter(X_test[:, 0], X_test[:, 1], c = score_p_x_y0_new)
#plt.colorbar()
#
#plt.figure()
#plt.scatter(X_test[:, 0], X_test[:, 1], c = score_p_x_y1_new)
#plt.colorbar()

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c = 1 - np.max(post_new, axis=1), vmin=0, vmax=.5, cmap = 'jet')
plt.colorbar()
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig("error_prob_mar.png")
plt.savefig("error_prob_mar.pdf", format='pdf')

#clf_mar = SVC(gamma='auto', probability=True)
#clf_mar.fit(X, Y_hat)
#
#post_new_mar = clf_mar.predict_proba(X_test)
#plt.figure(figsize=(6, 5))
#plt.scatter(X_test[:, 0], X_test[:, 1], c = 1 - np.max(post_new_mar, axis=1), vmin=0, vmax=.5, cmap = 'jet')
#plt.colorbar()
#plt.xlim([-10, 10])
#plt.ylim([-10, 10])

error_prob = np.zeros(N_test)
for i in range(N_test):
    if y_hat_mar_test[i] == 0:
        error_prob[i] = 1 - ((rv1.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]))/(rv1.pdf(X_test[i, :]) + 2*rv2.pdf(X_test[i, :]) + rv3.pdf(X_test[i, :])))
    elif y_hat_mar_test[i] == 1:
        error_prob[i] = 1 - ((rv3.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]))/(rv1.pdf(X_test[i, :]) + 2*rv2.pdf(X_test[i, :]) + rv3.pdf(X_test[i, :])))

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c = error_prob, vmin=0, vmax=.5, cmap = 'jet')
plt.colorbar()
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig("true_error_prob_mar.png")
plt.savefig("true_error_prob_mar.pdf", format='pdf')

score_mar = (1 - np.max(post_new, axis=1)) - error_prob

y_hat_optimal = np.zeros(N_test)
for i in range(N_test):
    if rv1.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]) > rv3.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]):
        y_hat_optimal[i] = 0
    elif rv1.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]) < rv3.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]):
        y_hat_optimal[i] = 1
    else:
        y_hat_optimal[i] = np.random.choice([0, 1], size=1, p=[0.5, 0.5], replace=False)


error_prob_optimal = np.zeros(N_test)
for i in range(N_test):
    if y_hat_optimal[i] == 0:
        error_prob_optimal[i] = 1 - ((rv1.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]))/(rv1.pdf(X_test[i, :]) + 2*rv2.pdf(X_test[i, :]) + rv3.pdf(X_test[i, :])))
    elif y_hat_optimal[i] == 1:
        error_prob_optimal[i] = 1 - ((rv3.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]))/(rv1.pdf(X_test[i, :]) + 2*rv2.pdf(X_test[i, :]) + rv3.pdf(X_test[i, :])))

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c = error_prob_optimal, vmin=0, vmax=.5, cmap = 'jet')
plt.colorbar()
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig("true_error_prob_optimal.png")
plt.savefig("true_error_prob_optimal.pdf", format='pdf')


# =============================================================================
# MCAR
# =============================================================================

index_unlabel = np.where(L == 0)[0]
post = np.zeros((np.size(L[L == 0]), 2))
y_hat_x_l0 = np.zeros((np.size(L[L == 0])))
x_l0 = X[L == 0]

for i in range(np.size(index_unlabel)):

    post[i, 0] = np.exp(score_p_x_y0_l0[i])/(np.exp(score_p_x_y0_l0[i]) + np.exp(score_p_x_y1_l0[i]))
    post[i, 1] = np.exp(score_p_x_y1_l0[i])/(np.exp(score_p_x_y0_l0[i]) + np.exp(score_p_x_y1_l0[i]))

            
    if post[i, 0] > post[i, 1]:    
        y_hat_x_l0[i] = 0 
    else:
        y_hat_x_l0[i] = 1

Y_hat = Y
Y_hat[L == 0] = y_hat_x_l0

plt.figure(figsize=(5, 5))
plt.scatter(X[Y_hat == 0, 0], X[Y_hat == 0, 1], marker='o', edgecolors='tab:blue', s=100, facecolors='none', label="0")
plt.scatter(X[Y_hat == 1, 0], X[Y_hat == 1, 1], marker='x', c='tab:orange', s=100, label="1") 
plt.legend()   
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig("data_self_label_mcar.png")
plt.savefig("data_self_label_mcar.pdf", format='pdf')

p_x_y0_new_mcar = BayesianGaussianMixture(n_components=10)
p_x_y0_new_mcar.fit(X[Y_hat == 0, :])

p_x_y1_new_mcar = BayesianGaussianMixture(n_components=10)
p_x_y1_new_mcar.fit(X[Y_hat == 1, :])

#N_test = 10000
#X_test, Y_test, L_test = functions.data_gen(N_test, 0, p_component, mean, cov)

#score_p_x_y0_new_mcar = p_x_y0_new_mcar.score_samples(X_test)
#score_p_x_y1_new_mcar = p_x_y1_new_mcar.score_samples(X_test)
#
#
#
#post_new_mcar = np.zeros((np.size(L_test[L_test == 0]), 2))
#for i in range(N_test):
#    post_new_mcar[i, 0] = (score_p_x_y0_new_mcar[i])/((score_p_x_y0_new_mcar[i]) + (score_p_x_y1_new_mcar[i]))
#    post_new_mcar[i, 1] = (score_p_x_y1_new_mcar[i])/((score_p_x_y0_new_mcar[i]) + (score_p_x_y1_new_mcar[i]))
#    
#    
##plt.figure()
##plt.scatter(X_test[:, 0], X_test[:, 1], c = score_p_x_y0_new_mcar)
##plt.colorbar()
##
##plt.figure()
##plt.scatter(X_test[:, 0], X_test[:, 1], c = score_p_x_y1_new_mcar)
##plt.colorbar()
#
#plt.figure(figsize=(6, 5))
#plt.scatter(X_test[:, 0], X_test[:, 1], c = 1 - np.max(post_new_mcar, axis=1), vmin=0, vmax=.5, cmap = 'jet')
#plt.colorbar()
#plt.xlim([-10, 10])
#plt.ylim([-10, 10])

clf = SVC(gamma='auto', probability=True)
clf.fit(X, Y_hat)

post_new_mcar = clf.predict_proba(X_test)
y_hat_mcar_test = np.argmax(post_new_mcar, axis=1)

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c = 1 - np.max(post_new_mcar, axis=1), vmin=0, vmax=.5, cmap = 'jet')
plt.colorbar()
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig("error_prob_mcar.png")
plt.savefig("error_prob_mcar.pdf", format='pdf')

error_prob_mcar = np.zeros(N_test)
for i in range(N_test):
    if y_hat_mcar_test[i] == 0:
        error_prob_mcar[i] = 1 - ((rv1.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]))/(rv1.pdf(X_test[i, :]) + 2*rv2.pdf(X_test[i, :]) + rv3.pdf(X_test[i, :])))
    elif y_hat_mcar_test[i] == 1:
        error_prob_mcar[i] = 1 - ((rv3.pdf(X_test[i, :]) + rv2.pdf(X_test[i, :]))/(rv1.pdf(X_test[i, :]) + 2*rv2.pdf(X_test[i, :]) + rv3.pdf(X_test[i, :])))

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c = error_prob_mcar, vmin=0, vmax=.5, cmap = 'jet')
plt.colorbar()
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig("true_error_prob_mcar.png")
plt.savefig("true_error_prob_mcar.pdf", format='pdf')

score_mcar = (1 - np.max(post_new_mcar, axis=1)) - error_prob_mcar

#brier_mar = brier_score_loss(Y_test, post_new[:, 0])
#brier_mcar = brier_score_loss(Y_test, post_new_mcar[:, 0])

#brier_score_mcar = np.mean(np.square(score_mcar))
#brier_score_mar = np.mean(np.square(score_mar))

brier_score_mcar = np.mean(np.square((1 - np.max(post_new_mcar, axis=1)) - [y_hat_mcar_test != Y_test]))
brier_score_mar = np.mean(np.square((1 - np.max(post_new, axis=1)) - [y_hat_mar_test != Y_test]))
brier_score_true_mcar = np.mean(np.square(error_prob_mcar - [y_hat_mcar_test != Y_test]))
brier_score_true_mar = np.mean(np.square(error_prob - [y_hat_mar_test != Y_test]))

bins = np.arange(0, .6, step=0.1)
hist_estimate_mcar, _, = np.histogram(1 - np.max(post_new_mcar, axis=1), bins = bins)
hist_true_mcar, _, = np.histogram(error_prob_mcar, bins = bins)

hist_estimate_mar, _, = np.histogram(1 - np.max(post_new, axis=1), bins = bins)
hist_true_mar, _, = np.histogram(error_prob, bins = bins)

#plt.figure(figsize=(5, 5))
#plt.plot(bins[1:], hist_true_mar/hist_estimate_mar, 'k', label = 'All')
#plt.grid()
##plt.xticks(np.arange(0, 1.2, step=0.2))
##plt.yticks(np.arange(0, 1.2, step=0.2))
#plt.xlabel(r"Estimated")
#plt.ylabel(r"Empirical")
#plt.tight_layout()

plt.figure()
plt.boxplot([score_mar, score_mcar], showfliers=False)
plt.xticks([1, 2], ['MAR', 'MCAR'])
plt.ylabel(r'$\hat{\pi}(\bm{x}) - \pi(\bm{x})$')

plt.figure(figsize=(6, 5))
plt.hist([score_mar, score_mcar], bins=[-.5, -.3, -.1, .1])
#plt.xticks([1, 2], ['MAR', 'MCAR'])
plt.xlabel(r'$\hat{\pi}(\bm{x}) - \pi(\bm{x})$')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("histogram_brier.png")
plt.savefig("histogram_brier.pdf", format='pdf')

plt.figure(figsize=(6, 5))
plt.hist([score_mar, score_mcar], bins=20)
#plt.xticks([1, 2], ['MAR', 'MCAR'])
plt.xlabel(r'$\hat{\pi}(\bm{x}) - \pi(\bm{x})$')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("histogram_brier.png")
plt.savefig("histogram_brier.pdf", format='pdf')