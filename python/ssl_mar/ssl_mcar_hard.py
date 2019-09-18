# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:14:51 2019

@author: Administrator
"""

from keras.datasets import mnist

import data as data
import model as model
import matplotlib.pyplot as plt

import numpy as np

from sklearn import mixture

plt.rcParams.update({'font.size': 20})

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
num_classes = 10
original_dim = image_size * image_size

x_train = data.vectorize_and_normalize(x_train, original_dim)
x_test = data.vectorize_and_normalize(x_test, original_dim)

# Load model
encoder, decoder = model.my_model(original_dim)
batch_size = 128

# All train data
z_train, _, _ = encoder.predict(x_train, batch_size=batch_size)
z_test, _, _ = encoder.predict(x_test, batch_size=batch_size)

# Labeling process 
num_labeled = 1000
leftout_classes = [0, 1, 7]

iteration_max = 100

ratio_1 = np.zeros((iteration_max, 19))
ratio_2 = np.zeros((iteration_max, 19))
ratio = np.zeros((iteration_max, 19))

for iteration in range(iteration_max):
    #z_labeled, y_labeled, z_unlabeled, y_unlabeled = data.labeling_process_1(z_train, y_train, num_labeled)
    z_labeled, y_labeled, z_unlabeled, y_unlabeled = data.labeling_process_2(z_train, y_train, num_labeled, leftout_classes)
    #z_labeled, y_labeled, z_unlabeled, y_unlabeled = data.labeling_process_3(z_train, y_train, num_classes, [5, 5, 100, 100, 100, 100, 100, 5, 100, 100])
    
    # GMM for unlabeled data and labeled data of each class
    pz_unlabeled = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_unlabeled)
    #pz_labeled_0 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==0)[0], :])
    #pz_labeled_1 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==1)[0], :])
    pz_labeled_2 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==2)[0], :])
    pz_labeled_3 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==3)[0], :])
    pz_labeled_4 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==4)[0], :])
    pz_labeled_5 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==5)[0], :])
    pz_labeled_6 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==6)[0], :])
    #pz_labeled_7 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==7)[0], :])
    pz_labeled_8 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==8)[0], :])
    pz_labeled_9 = mixture.BayesianGaussianMixture(n_components=5, max_iter = 500).fit(z_labeled[np.nonzero(y_labeled==9)[0], :])
    
    # Prior p(y), uniform or empirical
    py_prior = np.ones([10])*.1  
    py_labeled = np.zeros([10])
    py_labeled[0] = np.nonzero(y_labeled == 0)[0].shape[0]/y_labeled.shape[0]
    py_labeled[1] = np.nonzero(y_labeled == 1)[0].shape[0]/y_labeled.shape[0]
    py_labeled[2] = np.nonzero(y_labeled == 2)[0].shape[0]/y_labeled.shape[0]
    py_labeled[3] = np.nonzero(y_labeled == 3)[0].shape[0]/y_labeled.shape[0]
    py_labeled[4] = np.nonzero(y_labeled == 4)[0].shape[0]/y_labeled.shape[0]
    py_labeled[5] = np.nonzero(y_labeled == 5)[0].shape[0]/y_labeled.shape[0]
    py_labeled[6] = np.nonzero(y_labeled == 6)[0].shape[0]/y_labeled.shape[0]
    py_labeled[7] = np.nonzero(y_labeled == 7)[0].shape[0]/y_labeled.shape[0]
    py_labeled[8] = np.nonzero(y_labeled == 8)[0].shape[0]/y_labeled.shape[0]
    py_labeled[9] = np.nonzero(y_labeled == 9)[0].shape[0]/y_labeled.shape[0]
    
    # Log-lilkelihood: p(l = 0 | z) and p(y | z) for all unlabeled z
    ll_null = pz_unlabeled.score_samples(z_unlabeled)
    ll_y = np.zeros([ll_null.shape[0], 10])
    #ll_y[:, 0] = pz_labeled_0.score_samples(z_unlabeled)
    #ll_y[:, 1] = pz_labeled_1.score_samples(z_unlabeled)
    ll_y[:, 2] = pz_labeled_2.score_samples(z_unlabeled)
    ll_y[:, 3] = pz_labeled_3.score_samples(z_unlabeled)
    ll_y[:, 4] = pz_labeled_4.score_samples(z_unlabeled)
    ll_y[:, 5] = pz_labeled_5.score_samples(z_unlabeled)
    ll_y[:, 6] = pz_labeled_6.score_samples(z_unlabeled)
    #ll_y[:, 7] = pz_labeled_7.score_samples(z_unlabeled)
    ll_y[:, 8] = pz_labeled_8.score_samples(z_unlabeled)
    ll_y[:, 9] = pz_labeled_9.score_samples(z_unlabeled)
    
    ll_y[:, leftout_classes] = np.NINF
    
    # LRT 
    lrt_y = np.zeros([ll_null.shape[0], 10])
    for i in range(10):
        lrt_y[:, i] = ll_y[:, i]-ll_null
    
    # Sampling from the posterior
#    y_unlabeled_hat = data.sampling_posterior_1(ll_y)
#    y_unlabeled_hat = data.sampling_posterior_2(ll_y, lrt_y)
    y_unlabeled_hat, _ = data.mle(ll_y)
    #my_plt.plot_samples(z_unlabeled, y_unlabeled_hat, y_unlabeled)
    
    # GMM for labeled data + samples from posterior of each class
    z_0 = np.vstack((z_labeled[np.nonzero(y_labeled==0)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==0)[0], :]))
    z_1 = np.vstack((z_labeled[np.nonzero(y_labeled==1)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==1)[0], :]))
    z_2 = np.vstack((z_labeled[np.nonzero(y_labeled==2)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==2)[0], :]))
    z_3 = np.vstack((z_labeled[np.nonzero(y_labeled==3)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==3)[0], :]))
    z_4 = np.vstack((z_labeled[np.nonzero(y_labeled==4)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==4)[0], :]))
    z_5 = np.vstack((z_labeled[np.nonzero(y_labeled==5)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==5)[0], :]))
    z_6 = np.vstack((z_labeled[np.nonzero(y_labeled==6)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==6)[0], :]))
    z_7 = np.vstack((z_labeled[np.nonzero(y_labeled==7)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==7)[0], :]))
    z_8 = np.vstack((z_labeled[np.nonzero(y_labeled==8)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==8)[0], :]))
    z_9 = np.vstack((z_labeled[np.nonzero(y_labeled==9)[0], :], z_unlabeled[np.nonzero(y_unlabeled_hat==9)[0], :]))
    
    #pz_labeled_0_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_0)
    #pz_labeled_1_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_1)
    pz_labeled_2_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_2)
    pz_labeled_3_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_3)
    pz_labeled_4_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_4)
    pz_labeled_5_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_5)
    pz_labeled_6_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_6)
    #pz_labeled_7_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_7)
    pz_labeled_8_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_8)
    pz_labeled_9_new = mixture.BayesianGaussianMixture(n_components=5, max_iter = 1000).fit(z_9)
    
    # For test data
    # Log-likelihood
    ll_y_test = np.zeros([y_test.shape[0], 10])
    #ll_y_test[:, 0] = pz_labeled_0_new.score_samples(z_test)
    #ll_y_test[:, 1] = pz_labeled_1_new.score_samples(z_test)
    ll_y_test[:, 2] = pz_labeled_2_new.score_samples(z_test)
    ll_y_test[:, 3] = pz_labeled_3_new.score_samples(z_test)
    ll_y_test[:, 4] = pz_labeled_4_new.score_samples(z_test)
    ll_y_test[:, 5] = pz_labeled_5_new.score_samples(z_test)
    ll_y_test[:, 6] = pz_labeled_6_new.score_samples(z_test)
    #ll_y_test[:, 7] = pz_labeled_7_new.score_samples(z_test)
    ll_y_test[:, 8] = pz_labeled_8_new.score_samples(z_test)
    ll_y_test[:, 9] = pz_labeled_9_new.score_samples(z_test)
    
    ll_y_test[:, leftout_classes] = np.NINF
    
    y_test_mle, pe_test_mle = data.mle(ll_y_test)
    
    index_errors_1 = np.nonzero(np.logical_and(y_test != y_test_mle, np.isin(y_test, leftout_classes)))
    index_errors_2 = np.nonzero(np.logical_and(y_test != y_test_mle, np.logical_not(np.isin(y_test, leftout_classes)))) 
    
    index_1 = np.nonzero(np.isin(y_test, leftout_classes))
    index_2 = np.nonzero(np.logical_not(np.isin(y_test, leftout_classes)))
    
    bins = np.arange(0, 1, step=0.05)
    
    hist_error_1, _, = np.histogram(pe_test_mle[index_errors_1[0]], bins = bins)
    hist_1, _, = np.histogram(pe_test_mle[index_1[0]], bins = bins)
    
    hist_error_2, _, = np.histogram(pe_test_mle[index_errors_2[0]], bins = bins)
    hist_2, _, = np.histogram(pe_test_mle[index_2[0]], bins = bins)
    
    ratio_1[iteration, :] = hist_error_1/(hist_1 + hist_2)
    ratio_2[iteration, :] = hist_error_2/(hist_1 + hist_2)
    ratio[iteration, :] = hist_error_1/(hist_1 + hist_2) + hist_error_2/(hist_1 + hist_2)
    
    print(iteration)

# =============================================================================
# Visualization
# =============================================================================
    
ratio_1_average = np.mean(ratio_1, axis=0)
ratio_2_average = np.mean(ratio_2, axis=0)
ratio_average = np.mean(ratio, axis=0)

plt.figure(figsize=(6, 5))
plt.scatter(z_test[:, 0], z_test[:, 1], c=pe_test_mle, marker='x', cmap='jet', vmin = 0, vmax = .9)
plt.colorbar()
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.tight_layout()
plt.savefig("estimated_pe_mcar_hard.png")
plt.savefig("estimated_pe_mcar_hard.pdf", format='pdf')

plt.figure(figsize=(5, 5))
plt.scatter(bins[1:], ratio_1_average, marker= '+', label = 'Unobserved')
plt.scatter(bins[1:], ratio_2_average, marker= 'o', label = 'Observed')
plt.plot(bins[1:], ratio_average, 'k-', label = 'All')
plt.grid()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(np.arange(0, 1.2, step=0.2))
plt.yticks(np.arange(0, 1.2, step=0.2))
plt.xlabel("Estimated error probability")
plt.ylabel("Empirical error probability")
plt.tight_layout()
plt.savefig("empirical_pe_mcar_hard.png")
plt.savefig("empirical_pe_mcar_hard.pdf", format='pdf')