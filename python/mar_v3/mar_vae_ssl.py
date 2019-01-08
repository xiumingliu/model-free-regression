# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 13:29:31 2018

@author: Administrator
"""

'''
Using varational auto-encoder for semi-supervised learning under MAR assumption.
Dataset: MNIST

'''
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
#from scipy import linalg
#import argparse
#import os



#def vectorize_and_normalize(x, original_dim, num_classes):
#    for i in range(num_classes):
#        x[i] = np.reshape(x[i], [-1, original_dim])
#        x[i] = x[i].astype('float32') / 255
#    return x
    
def vectorize_and_normalize(x, original_dim):
    x = np.reshape(x, [-1, original_dim])
    x = x.astype('float32') / 255
    return x

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]


num_classes = 10

original_dim = image_size * image_size
x_train = vectorize_and_normalize(x_train, original_dim)
x_test = vectorize_and_normalize(x_test, original_dim)

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
encoded = Dense(128, activation='relu')(inputs)
x = Dense(intermediate_dim, activation='relu')(encoded)
encoded = Dense(intermediate_dim, activation='relu')(encoded)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
decoded  = Dense(128, activation='relu')(latent_inputs)
x = Dense(intermediate_dim, activation='relu')(decoded)
decoded = Dense(intermediate_dim, activation='relu')(decoded)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
models = (encoder, decoder)

# VAE loss = mse_loss or xent_loss + kl_loss
reconstruction_loss = binary_crossentropy(inputs, outputs)

reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
vae.load_weights('vae_mlp_mnist.h5')

# Discrete colormap
# define the colormap
cmap = cm.get_cmap("jet", 10)
bounds = np.linspace(0, 9, 10)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# All data
z_mean_train, _, _ = encoder.predict(x_train, batch_size=batch_size)
fig = plt.figure(figsize=(6, 5))
plt.scatter(z_mean_train[:, 0], z_mean_train[:, 1], c=y_train, cmap=cmap, vmin = -0.5, vmax = 9.5)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
plt.title('All training data')
plt.savefig("all_data.png")

# display a 30x30 2D manifold of digits
n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = n * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.imshow(figure, cmap='Greys_r')
#plt.savefig(filename)
plt.show()

# Discrete colormap
# define the colormap
cmap = cm.get_cmap("jet", 10)
bounds = np.linspace(0, 9, 10)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Labeling process (MAR)
# Only z[:, 0]^2 + z[:, 1]^2 > r^2 get labels, where r = 3
# Random labeling process according to z
def labeling_process(z_mean):
#    pval = (np.abs(z_mean[:, 0]) + np.abs(z_mean[:, 1]))/(np.abs(z_mean[:, 0]) + np.abs(z_mean[:, 1]) + 5000)
#    pval = np.ones([z_mean.shape[0]]) - np.ones([z_mean.shape[0]])/(np.abs(z_mean[:, 0]) + np.abs(z_mean[:, 1]))
    pval = np.ones([z_mean.shape[0]])/(1 + np.exp(-2*(np.sqrt(np.abs(z_mean[:, 0])**2 + np.abs(z_mean[:, 1])**2) - 3)))
    return pval

index_labeled = np.nonzero(np.random.binomial(1, labeling_process(z_mean_train)))[0]
index_unlabeled = np.setdiff1d(np.arange(z_mean_train.shape[0]), index_labeled)
z_mean_labeled = z_mean_train[index_labeled, :]
y_train_labeled = y_train[index_labeled]
z_mean_unlabeled = z_mean_train[index_unlabeled, :]
y_train_unlabeled = y_train[index_unlabeled]

#z_mean_labeled = z_mean_train[np.nonzero((z_mean_train[:, 0]**2 + z_mean_train[:, 1]**2) >= 9*np.ones(z_mean_train.shape[0]))[0], :]
#y_train_labeled = y_train[np.nonzero((z_mean_train[:, 0]**2 + z_mean_train[:, 1]**2) >= 9*np.ones(z_mean_train.shape[0]))[0]]
#z_mean_unlabeled = z_mean_train[np.nonzero((z_mean_train[:, 0]**2 + z_mean_train[:, 1]**2) < 9*np.ones(z_mean_train.shape[0]))[0], :]
#y_train_unlabeled = y_train[np.nonzero((z_mean_train[:, 0]**2 + z_mean_train[:, 1]**2) < 9*np.ones(z_mean_train.shape[0]))[0]]

fig = plt.figure(figsize=(6, 5))
plt.scatter(z_mean_labeled[:, 0], z_mean_labeled[:, 1], c=y_train_labeled, cmap=cmap, vmin = -0.5, vmax = 9.5)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
plt.title('Labeled data')
plt.savefig("labeled_data.png")

def plot_nll(z_mean, y, classK, cmap):
    z0 = np.linspace(-6., 6.)
    z1 = np.linspace(-6., 6.)
    Z0, Z1 = np.meshgrid(z0, z1)
    ZZ = np.array([Z0.ravel(), Z1.ravel()]).T
    if classK == -1:
#            z_mean_unlabeled, _, _ = encoder.predict(x_train_unlabeled, batch_size=batch_size)
        gmm = mixture.GaussianMixture(n_components=5, covariance_type='full', max_iter=500, tol=1e-3).fit(z_mean)
        nll = -gmm.score_samples(ZZ)
        nll = nll.reshape(Z0.shape)
        plt.figure(figsize=(6, 5))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=(0,0,0), alpha=0.05)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
        plt.colorbar()           
        plt.title('Negative log-likelihood: unlabeled')
#            plt.show()
        
    elif classK == -2: 
#            z_mean_labeled, _, _ = encoder.predict(x_train_labeled, batch_size=batch_size)
        gmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full', max_iter=500, tol=1e-3).fit(z_mean)
        nll = -gmm.score_samples(ZZ)
        nll = nll.reshape(Z0.shape)
        plt.figure(figsize=(6, 5))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y, cmap=cmap)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
        plt.colorbar()
        plt.title('Negative log-likelihood: labeled')
#            plt.show()
        
    else:    
        z_mean_classK = z_mean[np.nonzero(y == classK)[0], :]
        gmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full', max_iter=500, tol=1e-3).fit(z_mean_classK)           
        nll = -gmm.score_samples(ZZ)
        nll = nll.reshape(Z0.shape)
#        plt.figure(figsize=(6, 5))
#        plt.scatter(z_mean_classK[:, 0], z_mean_classK[:, 1], c=cmap(classK/10), cmap=cmap)
#        plt.xlabel("z[0]")
#        plt.ylabel("z[1]")
#        plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
#        plt.colorbar()
#        plt.title('Negative log-likelihood: '+str(classK))
#            plt.show()
    
    return gmm

pz_unlabeled = plot_nll(z_mean_unlabeled, None, -1, cmap)
pz_labeled = plot_nll(z_mean_labeled, y_train_labeled, -2, cmap)
pz_labeled_class0 = plot_nll(z_mean_labeled, y_train_labeled, 0, cmap)
pz_labeled_class1 = plot_nll(z_mean_labeled, y_train_labeled, 1, cmap)
pz_labeled_class2 = plot_nll(z_mean_labeled, y_train_labeled, 2, cmap)
pz_labeled_class3 = plot_nll(z_mean_labeled, y_train_labeled, 3, cmap)
pz_labeled_class4 = plot_nll(z_mean_labeled, y_train_labeled, 4, cmap)
pz_labeled_class5 = plot_nll(z_mean_labeled, y_train_labeled, 5, cmap)
pz_labeled_class6 = plot_nll(z_mean_labeled, y_train_labeled, 6, cmap)
pz_labeled_class7 = plot_nll(z_mean_labeled, y_train_labeled, 7, cmap)
pz_labeled_class8 = plot_nll(z_mean_labeled, y_train_labeled, 8, cmap)
pz_labeled_class9 = plot_nll(z_mean_labeled, y_train_labeled, 9, cmap)
#
py_labeled = np.zeros([10])
py_labeled[0] = np.nonzero(y_train_labeled == 0)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[1] = np.nonzero(y_train_labeled == 1)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[2] = np.nonzero(y_train_labeled == 2)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[3] = np.nonzero(y_train_labeled == 3)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[4] = np.nonzero(y_train_labeled == 4)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[5] = np.nonzero(y_train_labeled == 5)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[6] = np.nonzero(y_train_labeled == 6)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[7] = np.nonzero(y_train_labeled == 7)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[8] = np.nonzero(y_train_labeled == 8)[0].shape[0]/y_train_labeled.shape[0]
py_labeled[9] = np.nonzero(y_train_labeled == 9)[0].shape[0]/y_train_labeled.shape[0]

#ll_1 = pz_labeled.score_samples(z_mean_unlabeled)
#
#ll_0 = pz_unlabeled.score_samples(z_mean_unlabeled)
#lr = ll_1 - ll_0


ll0 = pz_labeled_class0.score_samples(z_mean_unlabeled)
ll1 = pz_labeled_class1.score_samples(z_mean_unlabeled)
ll2 = pz_labeled_class2.score_samples(z_mean_unlabeled)
ll3 = pz_labeled_class3.score_samples(z_mean_unlabeled)
ll4 = pz_labeled_class4.score_samples(z_mean_unlabeled)
ll5 = pz_labeled_class5.score_samples(z_mean_unlabeled)
ll6 = pz_labeled_class6.score_samples(z_mean_unlabeled)
ll7 = pz_labeled_class7.score_samples(z_mean_unlabeled)
ll8 = pz_labeled_class8.score_samples(z_mean_unlabeled)
ll9 = pz_labeled_class9.score_samples(z_mean_unlabeled)

ll_null = pz_unlabeled.score_samples(z_mean_unlabeled)

lr = np.zeros([10, z_mean_unlabeled.shape[0]])

for i in range(z_mean_unlabeled.shape[0]):
    lr[0, i] = ll0[i] - ll_null[i]
    lr[1, i] = ll1[i] - ll_null[i]
    lr[2, i] = ll2[i] - ll_null[i]
    lr[3, i] = ll3[i] - ll_null[i]
    lr[4, i] = ll4[i] - ll_null[i]
    lr[5, i] = ll5[i] - ll_null[i]
    lr[6, i] = ll6[i] - ll_null[i]
    lr[7, i] = ll7[i] - ll_null[i]
    lr[8, i] = ll8[i] - ll_null[i]
    lr[9, i] = ll9[i] - ll_null[i]
    
lr_max = np.max(lr, axis=0)

plt.figure(figsize=(6, 5))
plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=lr_max, cmap='jet', vmin = -1, vmax  =1)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.colorbar()
plt.title('Log likelihood ratio test for unlabeled')

py_prior = np.ones([10])*.1

def assign_labeles_map_soft(lr, z_mean_unlabeled, py_labeled, num_classes):
    index_similar_features = np.nonzero(lr_max > 0)
    index_unfamiliar_features = np.setdiff1d(np.arange(lr_max.shape[0]), index_similar_features)
    post = np.zeros([10, lr_max.shape[0]])
#    post[0, index_similar_features[0]] = np.exp(pz_labeled_class0.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[0]
#    post[1, index_similar_features[0]] = np.exp(pz_labeled_class1.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[1]
#    post[2, index_similar_features[0]] = np.exp(pz_labeled_class2.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[2]
#    post[3, index_similar_features[0]] = np.exp(pz_labeled_class3.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[3]
#    post[4, index_similar_features[0]] = np.exp(pz_labeled_class4.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[4]
#    post[5, index_similar_features[0]] = np.exp(pz_labeled_class5.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[5]
#    post[6, index_similar_features[0]] = np.exp(pz_labeled_class6.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[6]
#    post[7, index_similar_features[0]] = np.exp(pz_labeled_class7.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[7]
#    post[8, index_similar_features[0]] = np.exp(pz_labeled_class8.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[8]
#    post[9, index_similar_features[0]] = np.exp(pz_labeled_class9.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[9]
    
    post[0, index_similar_features[0]] = np.exp(pz_labeled_class0.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[0]
    post[1, index_similar_features[0]] = np.exp(pz_labeled_class1.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[1]
    post[2, index_similar_features[0]] = np.exp(pz_labeled_class2.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[2]
    post[3, index_similar_features[0]] = np.exp(pz_labeled_class3.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[3]
    post[4, index_similar_features[0]] = np.exp(pz_labeled_class4.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[4]
    post[5, index_similar_features[0]] = np.exp(pz_labeled_class5.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[5]
    post[6, index_similar_features[0]] = np.exp(pz_labeled_class6.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[6]
    post[7, index_similar_features[0]] = np.exp(pz_labeled_class7.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[7]
    post[8, index_similar_features[0]] = np.exp(pz_labeled_class8.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[8]
    post[9, index_similar_features[0]] = np.exp(pz_labeled_class9.score_samples(z_mean_unlabeled[index_similar_features[0], :]))*py_prior[9]
    
    post[:, index_unfamiliar_features] = py_prior[0]
    
    post_labeles_soft = np.zeros(lr_max.shape[0])
    for i in range(lr_max.shape[0]):
        pvals = post[:, i]/np.sum(post[:, i])
        post_labeles_soft[i] = np.nonzero(np.random.multinomial(1, pvals))[0][0]
#    post_labeles_soft = np.argmax(post, axis=0)
    return index_similar_features[0], post_labeles_soft

# Post sampling
index_similar_features, map_labeles_soft = assign_labeles_map_soft(lr, z_mean_unlabeled, py_labeled, num_classes)
#    index_unfamiliar_features = np.setdiff1d(np.arange(lr.shape[0]), index_similar_features)
plt.figure(figsize=(6, 5))
#    plt.scatter(z_mean_unlabeled[index_unfamiliar_features, 0], z_mean_unlabeled[index_unfamiliar_features, 1], c=(0,0,0), alpha=0.05)
plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=map_labeles_soft, cmap=cmap, vmin = -0.5, vmax = 9.5)
plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.title('Posterior sampling for unlabeled training data')
plt.savefig("post_sample_unlabeled.png")

index_errors = np.nonzero(y_train_unlabeled!= map_labeles_soft)
plt.figure(figsize=(5, 5))
plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
plt.scatter(z_mean_unlabeled[index_errors, 0], z_mean_unlabeled[index_errors, 1], c='red', alpha=0.05)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.title('Posterior sampling errors: presentage ' +str(index_errors[0].shape[0]/y_train_unlabeled.shape[0]))
plt.savefig("post_sample_errors.png")

z_mean_unlabeled_new = z_mean_unlabeled
z_mean_labeled_new = np.vstack((z_mean_labeled, z_mean_unlabeled))
y_train_labeled_new = np.concatenate((y_train_labeled, map_labeles_soft.astype('int8')))
