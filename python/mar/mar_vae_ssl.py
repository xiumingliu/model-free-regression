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
import matplotlib.cm as cm
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
        num_labeled = np.random.randint(15, max_num_labeled)
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

num_classes = 10
x_train, y_train = split_by_class(x_train, y_train, 10)
x_test, y_test = split_by_class(x_test, y_test, 10)

max_num_labeled = 500
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


    # Discrete colormap
    # define the colormap
    cmap = cm.get_cmap("jet", 10)
    bounds = np.linspace(0, 9, 10)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # All data
    z_mean_labeled, _, _ = encoder.predict(np.vstack(x_train_labeled), batch_size=batch_size)
    z_mean_unlabeled, _, _ = encoder.predict(np.vstack(x_train_unlabeled), batch_size=batch_size)
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
    plt.scatter(z_mean_labeled[:, 0], z_mean_labeled[:, 1], c=np.concatenate(y_train_labeled), cmap=cmap, vmin = -0.5, vmax = 9.5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
#    plt.show()
    plt.title('All training data: labeled and unlabeled')
    plt.savefig("all_data.png")
    
    def plot_nll(x_train_labeled, classK, cmap):
        z0 = np.linspace(-6., 6.)
        z1 = np.linspace(-6., 6.)
        Z0, Z1 = np.meshgrid(z0, z1)
        ZZ = np.array([Z0.ravel(), Z1.ravel()]).T
        if classK == -1:
            z_mean_unlabeled, _, _ = encoder.predict(np.vstack(x_train_unlabeled), batch_size=batch_size)
            gmm = mixture.GaussianMixture(n_components=15, covariance_type='full', max_iter=200, tol=1e-3).fit(z_mean_unlabeled)
            nll = -gmm.score_samples(ZZ)
            nll = nll.reshape(Z0.shape)
            plt.figure(figsize=(6, 5))
            plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
            plt.colorbar()           
            plt.title('Negative log-likelihood: unlabeled')
#            plt.show()
            plt.savefig("gmm_unlabeled.png")
        elif classK == -2: 
            z_mean_labeled, _, _ = encoder.predict(np.vstack(x_train_labeled), batch_size=batch_size)
            gmm = mixture.BayesianGaussianMixture(n_components=15, covariance_type='full', max_iter=200, tol=1e-3).fit(z_mean_labeled)
            nll = -gmm.score_samples(ZZ)
            nll = nll.reshape(Z0.shape)
            z_mean_labeled, _, _ = encoder.predict(np.vstack(x_train_labeled), batch_size=batch_size)
            plt.figure(figsize=(6, 5))
            plt.scatter(z_mean_labeled[:, 0], z_mean_labeled[:, 1], c=np.concatenate(y_train_labeled), cmap=cmap)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
            plt.colorbar()
            plt.title('Negative log-likelihood: labeled')
#            plt.show()
            plt.savefig("gmm_labeled.png")
        else:    
            z_mean_labeled_classK, _, _ = encoder.predict(x_train_labeled[classK], batch_size=batch_size)
            gmm = mixture.BayesianGaussianMixture(n_components=15, covariance_type='full', max_iter=200, tol=1e-3).fit(z_mean_labeled_classK)           
            nll = -gmm.score_samples(ZZ)
            nll = nll.reshape(Z0.shape)
            plt.figure(figsize=(6, 5))
            plt.scatter(z_mean_labeled_classK[:, 0], z_mean_labeled_classK[:, 1], c=cmap(classK/10), cmap=cmap)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
            plt.colorbar()
            plt.title('Negative log-likelihood: '+str(classK))
#            plt.show()
            plt.savefig("gmm_labeled_class"+str(classK)+".png")
            
        return gmm
      
    pz_unlabeled = plot_nll(x_train_unlabeled, -1, cmap)
    pz_labeled = plot_nll(x_train_labeled, -2, cmap)
    pz_labeled_class0 = plot_nll(x_train_labeled, 0, cmap)
    pz_labeled_class1 = plot_nll(x_train_labeled, 1, cmap)
    pz_labeled_class2 = plot_nll(x_train_labeled, 2, cmap)
    pz_labeled_class3 = plot_nll(x_train_labeled, 3, cmap)
    pz_labeled_class4 = plot_nll(x_train_labeled, 4, cmap)
    pz_labeled_class5 = plot_nll(x_train_labeled, 5, cmap)
    pz_labeled_class6 = plot_nll(x_train_labeled, 6, cmap)
    pz_labeled_class7 = plot_nll(x_train_labeled, 7, cmap)
    pz_labeled_class8 = plot_nll(x_train_labeled, 8, cmap)
    pz_labeled_class9 = plot_nll(x_train_labeled, 9, cmap)
    
    num_y_train_labeled = 0
    for i in range(num_classes):
        num_y_train_labeled = num_y_train_labeled + y_train_labeled[i].shape[0] 
    
    py_labeled_class0 = y_train_labeled[0].shape[0]/num_y_train_labeled
    py_labeled_class1 = y_train_labeled[1].shape[0]/num_y_train_labeled
    py_labeled_class2 = y_train_labeled[2].shape[0]/num_y_train_labeled
    py_labeled_class3 = y_train_labeled[3].shape[0]/num_y_train_labeled
    py_labeled_class4 = y_train_labeled[4].shape[0]/num_y_train_labeled
    py_labeled_class5 = y_train_labeled[5].shape[0]/num_y_train_labeled
    py_labeled_class6 = y_train_labeled[6].shape[0]/num_y_train_labeled
    py_labeled_class7 = y_train_labeled[7].shape[0]/num_y_train_labeled
    py_labeled_class8 = y_train_labeled[8].shape[0]/num_y_train_labeled
    py_labeled_class9 = y_train_labeled[9].shape[0]/num_y_train_labeled
    
    def glrt(pz_labeled, pz_unlabeled, z_mean_unlabeled):
#        z_mean_unlabeled, _, _ = encoder.predict(np.vstack(x_train_unlabeled), batch_size=batch_size)
        ll_1 = pz_labeled.score_samples(z_mean_unlabeled)
        ll_0 = pz_unlabeled.score_samples(z_mean_unlabeled)
        lr = ll_1 - ll_0
        return lr
    
    lr = glrt(pz_labeled, pz_unlabeled, z_mean_unlabeled)
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=lr, cmap='jet')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.colorbar()
    plt.title('Log likelihood ratio test for unlabeled')
#    plt.show()    
    plt.savefig("LRT_unlabeled_data.png")
    
    def assign_labeles_mle(lr, z_mean_unlabeled, num_classes):
        index_similar_features = np.nonzero(lr > 0)
        ll = ll = np.zeros([10, index_similar_features[0].shape[0]])
        ll[0, :] = pz_labeled_class0.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[1, :] = pz_labeled_class1.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[2, :] = pz_labeled_class2.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[3, :] = pz_labeled_class3.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[4, :] = pz_labeled_class4.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[5, :] = pz_labeled_class5.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[6, :] = pz_labeled_class6.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[7, :] = pz_labeled_class7.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[8, :] = pz_labeled_class8.score_samples(z_mean_unlabeled[index_similar_features[0]])
        ll[9, :] = pz_labeled_class9.score_samples(z_mean_unlabeled[index_similar_features[0]])
        mle_labeles = np.argmax(ll, axis=0)
        return index_similar_features[0], mle_labeles
    
    index_similar_features, mle_labeles = assign_labeles_mle(lr, z_mean_unlabeled, num_classes)
    index_unfamiliar_features = np.setdiff1d(np.arange(lr.shape[0]), index_similar_features)
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_unlabeled[index_unfamiliar_features, 0], z_mean_unlabeled[index_unfamiliar_features, 1], c=(0,0,0), alpha=0.05)
    plt.scatter(z_mean_unlabeled[index_similar_features, 0], z_mean_unlabeled[index_similar_features, 1], c=mle_labeles, cmap=cmap, vmin = -0.5, vmax = 9.5)
    plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.title('MLE of unlabeled training data')
    plt.savefig("MLE_unlabeled.png")
    
    index_errors = np.nonzero(np.concatenate(y_train_unlabeled)[index_similar_features] != mle_labeles)
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
    plt.scatter(z_mean_unlabeled[index_errors, 0], z_mean_unlabeled[index_errors, 1], c='red', alpha=0.05)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.title('Labeling errors: presentage ' +str(index_errors[0].shape[0]/index_similar_features.shape[0]))
    plt.savefig("errors.png")
        
    
    
    
    
    
    
    