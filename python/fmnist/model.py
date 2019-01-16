# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:20:15 2019

@author: Administrator
"""

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

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


def my_model(original_dim):
    # network parameters
    input_shape = (original_dim, )
    intermediate_dim = 512
    latent_dim = 3
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    encoded = Dense(128, activation='relu')(inputs)
    x = Dense(intermediate_dim, activation='relu')(encoded)
    encoded = Dense(intermediate_dim, activation='relu')(encoded)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
#    inputs = Input(shape=input_shape, name='encoder_input')
#    encoded = Dense(128, activation='relu')(inputs)
#    x = Dense(intermediate_dim, activation='relu')(encoded)
#    encoded = Dense(intermediate_dim, activation='relu')(encoded)
#    encoded = Dense(intermediate_dim, activation='relu')(encoded)
#    z_mean = Dense(latent_dim, name='z_mean')(encoded)
#    z_log_var = Dense(latent_dim, name='z_log_var')(encoded)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#    encoder.summary()
#    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    decoded  = Dense(128, activation='relu')(latent_inputs)
    x = Dense(intermediate_dim, activation='relu')(decoded)
    decoded = Dense(intermediate_dim, activation='relu')(decoded)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    
    
#    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#    decoded  = Dense(intermediate_dim, activation='relu')(latent_inputs)
#    decoded = Dense(intermediate_dim, activation='relu')(decoded)
#    decoded = Dense(intermediate_dim, activation='relu')(decoded)
#    x = Dense(128, activation='relu')(decoded)
#    outputs = Dense(original_dim, activation='sigmoid')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
#    decoder.summary()
#    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
#    vae.summary()
#    plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
    vae.load_weights('vae_mlp_fmnist_d3.h5')
    
    return encoder, decoder

def my_model_train(x_train, x_test, original_dim):
    batch_size = 128
    latent_dim = 5
    epochs = 50
    
    # network parameters
    input_shape = (original_dim, )
    intermediate_dim = 512
    
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
#    encoder.summary()
#    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    decoded  = Dense(128, activation='relu')(latent_inputs)
    x = Dense(intermediate_dim, activation='relu')(decoded)
    decoded = Dense(intermediate_dim, activation='relu')(decoded)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
#    decoder.summary()
#    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
#    vae.summary()
#    plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
#    vae.load_weights('vae_mlp_mnist.h5')
    vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))
    vae.save_weights('vae_mlp_fmnist_d5.h5')
    
    return encoder, decoder