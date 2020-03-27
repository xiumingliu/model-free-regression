# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:01:46 2019

@author: Administrator
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape, concatenate
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.optimizers import Adam, SGD
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt

def generateRandomVectors(y_train):
    vectors = []
    labels = np.zeros((len(y_train), 11))
    labels[range(len(y_train)), np.array(y_train).astype(int)] = 1
    for index,y in enumerate(y_train):
        if y == 10:
            l = np.random.randint(0, 10)
        else:
            l = y
        mean = [10*np.cos((l*2*np.pi)/10), 10*np.sin((l*2*np.pi)/10)]
        v1 = [np.cos((l*2*np.pi)/10), np.sin((l*2*np.pi)/10)]
        v2 = [-np.sin((l*2*np.pi)/10), np.cos((l*2*np.pi)/10)]
        a1 = 8
        a2 = .4
        M =np.vstack((v1,v2)).T
        S = np.array([[a1, 0], [0, a2]])
        cov = np.dot(np.dot(M, S), np.linalg.inv(M))
        #cov = cov*cov.T
        vec = np.random.multivariate_normal(mean=mean, cov=cov,
                                            size=1)
        vectors.append(vec)
    return (np.array(vectors).reshape(-1, 2), labels)

def train(x_train, y_train, x_test, y_test, batch_size=32, epochs=10000, save_interval=500):
    for epoch in range(epochs):
        #---------------Train Discriminator -------------
        # Select a random half batch of images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]
        y_batch = y_train[idx]
        # Generate a half batch of new images
        latent_fake = self.encoder.predict(imgs)
        #gen_imgs = self.decoder.predict(latent_fake)
        #latent_real = np.random.normal(size=(half_batch, self.encoded_dim))
        (latent_real, labels) = self.generateRandomVectors(y_batch)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch([latent_real, labels], valid)
        d_loss_fake = self.discriminator.train_on_batch([latent_fake, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        #idx = np.random.randint(0, x_train.shape[0], batch_size)
        #imgs = x_train[idx]
        #y_batch = y_train[idx]
        #(_, labels) = self.generateRandomVectors(y_batch)
        # Generator wants the discriminator to label the generated representations as valid
        valid_y = np.ones((batch_size, 1))

        # Train the autoencode reconstruction
        g_loss_reconstruction = self.autoencoder.train_on_batch(imgs, imgs)

        # Train generator
        g_logg_similarity = self.encoder_discriminator.train_on_batch([imgs, labels], valid_y)
        # Plot the progress
        print ("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1],
               g_logg_similarity[1], g_loss_reconstruction))
        if(epoch % save_interval == 0):
            self.imagegrid(epoch)
            self.saveLatentMap(epoch, x_test, y_test)

# =============================================================================
# Load data
# =============================================================================

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
N_train = np.size(y_train)

# Unbalanced numbers of labeled data in different classes
num_labeled = np.array([100, 100, 100, 100, 100, 0, 0, 0, 0, 0])

index_labeled = []
for i in range(10): 
    index_labeled.append(np.random.choice(np.where(y_train == i)[0], num_labeled[i], replace=False))
index_labeled = np.concatenate(index_labeled)

x_train_labeled = x_train[index_labeled, :, :]
y_train_labeled = y_train[index_labeled]
x_train_unlabeld = x_train[np.setdiff1d(np.arange(N_train), index_labeled), :, :]
y_train_unlabeld = y_train[np.setdiff1d(np.arange(N_train), index_labeled)]

# =============================================================================
# AAE
# =============================================================================
img_shape=(28, 28)
encoded_dim=2

optimizer_reconst = Adam(0.0001)
optimizer_discriminator = Adam(0.0001)

# Encoder
encoder = Sequential()
encoder.add(Flatten(input_shape=img_shape))
encoder.add(Dense(1000, activation='relu'))
encoder.add(Dense(1000, activation='relu'))
encoder.add(Dense(encoded_dim))
encoder.summary()

# Decoder
decoder = Sequential()
decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim))
decoder.add(Dense(1000, activation='relu'))
decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
decoder.add(Reshape(img_shape))
decoder.summary()

# Discriminator
latent_input = Input(shape=(encoded_dim,))
labels_input = Input(shape=(11,))
concated = concatenate([latent_input, labels_input])
discriminator = Sequential()
discriminator.add(Dense(1000, activation='relu', input_dim=encoded_dim+11))
discriminator.add(Dense(1000, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))
out = discriminator(concated)
discriminator.summary()

img = Input(shape=img_shape)
label_code = Input(shape=(11,))

encoded_repr = encoder(img)
gen_img = decoder(encoded_repr)

# Instantiate models
autoencoder = Model(img, gen_img)
discriminator_model = Model([latent_input, labels_input], out)
valid = discriminator_model([encoded_repr, label_code])
encoder_discriminator = Model([img, label_code], valid)

discriminator_model.compile(optimizer=optimizer_discriminator, loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.compile(optimizer=optimizer_reconst, loss ='mse')

for layer in discriminator_model.layers:
    layer.trainable = False
encoder_discriminator.compile(optimizer=optimizer_discriminator, loss='binary_crossentropy', metrics=['accuracy'])



