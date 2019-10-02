# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import mixture
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.datasets import load_boston
pd.options.display.float_format = '{:,.2f}'.format

dataset = load_boston()

print("[INFO] keys : {}".format(dataset.keys()))
print("[INFO] features shape : {}".format(dataset.data.shape))
print("[INFO] target shape   : {}".format(dataset.target.shape))
print("[INFO] feature names")
print(dataset.feature_names)

print("[INFO] dataset summary")
print(dataset.DESCR)

df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
df["PRICE"] = dataset.target

print("PEARSON CORRELATION")
print(df.corr(method="pearson"))
sns.heatmap(df.corr(method="pearson"))
plt.savefig("heatmap_pearson.png")
#plt.clf()
#plt.close()

#x = dataset.data
#y = np.round(dataset.target)

x_train = dataset.data
y_train = dataset.target

x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255
y_train = y_train.astype('float32') / 255


#plt.hist(y, bins=np.arange(min(y), max(y) + 1, 1))

# network parameters
original_dim = 13
input_shape = (original_dim, )
intermediate_dim = 128
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
encoded = Dense(intermediate_dim, activation='relu')(inputs)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(intermediate_dim, activation='relu')(encoded)
z_mean = Dense(latent_dim, name='z_mean')(encoded)
z_log_var = Dense(latent_dim, name='z_log_var')(encoded)

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

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
decoded = Dense(intermediate_dim, activation='relu')(latent_inputs)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(intermediate_dim, activation='relu')(decoded)
outputs = Dense(original_dim, activation='sigmoid')(decoded)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')


models = (encoder, decoder)
#data = (x_test, y_test)

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
#plot_model(vae,
#           to_file='vae_mlp.png',
#           show_shapes=True)

#vae.load_weights()

# train the autoencoder
vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_train, None))
vae.save_weights('vae_mlp.h5')

z_mean, _, _ = encoder.predict(x_train, batch_size=batch_size)

plt.figure(figsize = (6, 5))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c = y_train)
plt.colorbar()


plt.figure(figsize = (6, 5))
plt.scatter(x_train[:, 5], x_train[:, 12], c = y_train)
plt.colorbar()