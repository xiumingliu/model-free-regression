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
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from scipy import linalg
import argparse
import os

# Split data into classes
def split_by_class(x, y, num_classes):
    x_result = [0]*num_classes
    y_result = [0]*num_classes
    for i in range(num_classes):
        idx_i = np.where(y == i)[0]
        x_result[i] = x[idx_i, :, :]
        y_result[i] = y[idx_i]
    return x_result, y_result

# Split training data into labeled and unlabeled for SSL, under MAR assumption
def missing_at_random(x, y, num_classes, max_num_labeled): 
    x_labeled = [0]*num_classes
    y_labeled = [0]*num_classes
    x_unlabeled = [0]*num_classes
    y_unlabeled = [0]*num_classes
    for i in range(num_classes):
        num_labeled = np.random.randint(1, max_num_labeled)
        x_labeled[i] = x[i][1:num_labeled]
        y_labeled[i] = y[i][1:num_labeled]
        x_unlabeled[i] = x[i][num_labeled+1:]
        y_unlabeled[i] = y[i][num_labeled+1:]
    return x_labeled, y_labeled, x_unlabeled, y_unlabeled

def vectorize_and_normalize(x, original_dim, num_classes):
    for i in range(num_classes):
        x[i] = np.reshape(x[i], [-1, original_dim])
        x[i] = x[i].astype('float32') / 255
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

x_train, y_train = split_by_class(x_train, y_train, 10)
x_test, y_test = split_by_class(x_test, y_test, 10)

max_num_labeled = 500;
x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled = missing_at_random(x_train, y_train, 10, max_num_labeled)

original_dim = image_size * image_size
x_train_labeled = vectorize_and_normalize(x_train_labeled, original_dim, 10)
x_train_unlabeled = vectorize_and_normalize(x_train_unlabeled, original_dim, 10)
x_test = vectorize_and_normalize(x_test, original_dim, 10)

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
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data_test = (np.vstack(x_test), np.concatenate(y_test))
    data_labeled = (np.vstack(x_train_labeled), np.concatenate(y_train_labeled))

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder using both labeled and unlabeled data
        vae.fit(np.vstack((np.vstack(x_train_labeled), np.vstack(x_train_unlabeled))), epochs=epochs, batch_size=batch_size, validation_data=(np.vstack(x_test), None))
        vae.save_weights('vae_mlp_mnist.h5')

    z_mean_labeled, _, _ = encoder.predict(np.vstack(x_train_labeled), batch_size=batch_size)
    z_mean_unlabeled, _, _ = encoder.predict(np.vstack(x_train_unlabeled), batch_size=batch_size)
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
    plt.scatter(z_mean_labeled[:, 0], z_mean_labeled[:, 1], c=np.concatenate(y_train_labeled), cmap='jet')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    
    # Unlabeled data 
    gmm_unlabeled = mixture.GaussianMixture(n_components=15, covariance_type='full', max_iter=200, tol=1e-3).fit(z_mean_unlabeled)
    
    z0 = np.linspace(-6., 6.)
    z1 = np.linspace(-6., 6.)
    Z0, Z1 = np.meshgrid(z0, z1)
    ZZ = np.array([Z0.ravel(), Z1.ravel()]).T
    nll = -gmm_unlabeled.score_samples(ZZ)
    nll = nll.reshape(Z0.shape)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
    plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
    plt.colorbar()
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.show()
    
    # Labeled 0
    z_mean_labeled_0, _, _ = encoder.predict(x_train_labeled[0], batch_size=batch_size)
    gmm_labeled_0 = mixture.GaussianMixture(n_components=15, covariance_type='full', max_iter=200, tol=1e-3).fit(z_mean_labeled_0)
    
    nll = -gmm_labeled_0.score_samples(ZZ)
    nll = nll.reshape(Z0.shape)
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_labeled_0[:, 0], z_mean_labeled_0[:, 1], c=(0,0,0), alpha=0.05)
    plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
    plt.colorbar()
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.show()