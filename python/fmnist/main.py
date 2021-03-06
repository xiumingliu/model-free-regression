# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:06:06 2019

@author: Administrator
"""

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import MaxPooling2D
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
#import os

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
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

def vectorize_and_normalize(x, original_dim):
    x = np.reshape(x, [-1, original_dim])
    x = x.astype('float32') / 255
    return x


num_classes = 10 #
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

image_size = x_train.shape[1]
channel = 1
num_classes = 10
original_dim = image_size * image_size * channel


x_train = vectorize_and_normalize(x_train, original_dim)
x_test = vectorize_and_normalize(x_test, original_dim)


# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 10
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
encoded = Dense(128, activation='relu')(inputs)
x = Dense(intermediate_dim, activation='relu')(encoded)
encoded = Dense(intermediate_dim, activation='relu')(encoded)
encoded = Dense(intermediate_dim, activation='relu')(encoded)
z_mean = Dense(latent_dim, name='z_mean')(encoded)
z_log_var = Dense(latent_dim, name='z_log_var')(encoded)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
decoded  = Dense(intermediate_dim, activation='relu')(latent_inputs)
decoded = Dense(intermediate_dim, activation='relu')(decoded)
decoded = Dense(intermediate_dim, activation='relu')(decoded)
x = Dense(128, activation='relu')(decoded)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

models = (encoder, decoder)
#    data_test = (np.vstack(x_test), np.concatenate(y_test))
data_train = (x_train, y_train)

# VAE loss = mse_loss or xent_loss + kl_loss

reconstruction_loss = mse(inputs, outputs)

#reconstruction_loss = binary_crossentropy(inputs, outputs)

reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae, to_file='vae_mlp.png', show_shapes=True)


vae.load_weights('vae_mlp_fmnist_d10.h5')

# train the autoencoder using both labeled and unlabeled data
#vae.fit(np.vstack((x_train, x_train)), epochs=epochs, batch_size=batch_size, validation_data=(np.vstack(x_test), None))
#vae.save_weights('vae_mlp_fmnist_d10.h5')

z_train, _, _ = encoder.predict(x_train, batch_size=batch_size)
plt.figure(figsize = (3, 3))
plt.imshow(decoder.predict(z_train[1000, :].reshape((1, latent_dim))).reshape((image_size, image_size)))

#plt.figure(figsize = (6, 5))
#plt.scatter(z_train[:, 0], z_train[:, 1], c = y_train)
#plt.colorbar()
