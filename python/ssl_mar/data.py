# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:16:49 2019

@author: Administrator
"""

import numpy as np

def vectorize_and_normalize(x, original_dim):
    x = np.reshape(x, [-1, original_dim])
    x = x.astype('float32') / 255
    return x

def labeling_process_1(z_train, y_train, num_labeled):
    # MCAR
    index_labeled = np.random.choice(z_train.shape[0], num_labeled)
    index_unlabeled = np.setdiff1d(np.arange(z_train.shape[0]), index_labeled)
    z_labeled = z_train[index_labeled, :]
    y_labeled = y_train[index_labeled]
    z_unlabeled = z_train[index_unlabeled, :]
    y_unlabeled = y_train[index_unlabeled]
    return z_labeled, y_labeled, z_unlabeled, y_unlabeled

def labeling_process_2(z_train, y_train, num_labeled, leftout_classes):
    # No labeled data in leftout_classes, MCAR in rest
    index_train = np.arange(y_train.shape[0])
    index_train = index_train[np.logical_not(np.in1d(y_train, leftout_classes))]    
    
    index_labeled = np.random.choice(index_train, num_labeled)
    index_unlabeled = np.setdiff1d(np.arange(z_train.shape[0]), index_labeled)
    
    z_labeled = z_train[index_labeled, :]
    y_labeled = y_train[index_labeled]
    z_unlabeled = z_train[index_unlabeled, :]
    y_unlabeled = y_train[index_unlabeled]
    return z_labeled, y_labeled, z_unlabeled, y_unlabeled

# Split data into classes
def split_by_class(z, y, num_classes):
    z_result = [0]*num_classes
    y_result = [0]*num_classes
    for i in range(num_classes):
        idx_i = np.where(y == i)[0]
        z_result[i] = z[idx_i, :]
        y_result[i] = y[idx_i]
    return z_result, y_result

def labeling_process_3(z, y, num_classes, num_labeled): 
    z, y = split_by_class(z, y, num_classes)
    z_labeled = [0]*num_classes
    y_labeled = [0]*num_classes
    z_unlabeled = [0]*num_classes
    y_unlabeled = [0]*num_classes
    for i in range(num_classes):
        this_index_labeled = np.random.choice(z[i].shape[0], num_labeled[i])
        this_index_unlabeled = np.setdiff1d(np.arange(z[i].shape[0]), this_index_labeled)
        
        z_labeled[i] = z[i][this_index_labeled, :]
        y_labeled[i] = y[i][this_index_labeled]
        z_unlabeled[i] = z[i][this_index_unlabeled, :]
        y_unlabeled[i] = y[i][this_index_unlabeled]
    z_labeled = np.vstack(z_labeled)
    y_labeled = np.concatenate(y_labeled)
    z_unlabeled = np.vstack(z_unlabeled)
    y_unlabeled = np.concatenate(y_unlabeled)
    return z_labeled, y_labeled, z_unlabeled, y_unlabeled

def sampling_posterior_1(ll): 
    # Uniform prior
    post = np.exp(ll)/(np.sum(np.exp(ll), axis = 1)[:, None])
    samples = np.zeros(ll.shape[0])
    for i in range(samples.shape[0]):
        samples[i] = np.nonzero(np.random.multinomial(1, post[i, :]))[0][0]
    return samples

def sampling_posterior_2(ll, lrt): 
    # Uniform prior
    post = np.exp(ll)/(np.sum(np.exp(ll), axis = 1)[:, None])
    samples = np.zeros(ll.shape[0])
    for i in range(samples.shape[0]):
        if np.max(lrt[i, :]) > 0:
            samples[i] = np.nonzero(np.random.multinomial(1, post[i, :]))[0][0]
        else:
            samples[i] = np.nonzero(np.random.multinomial(1, .1*np.ones([10])))[0][0]    
    return samples

def mle(ll):
    y_mle = np.argmax(ll, axis=1)
    post = np.exp(ll)/(np.sum(np.exp(ll), axis = 1)[:, None]) 
    pe = np.zeros(y_mle.shape[0])
    for i in range(y_mle.shape[0]):
        pe[i] = 1 - post[i, y_mle[i]]
    return y_mle, pe