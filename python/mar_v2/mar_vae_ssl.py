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
#from scipy import linalg
import argparse
#import os

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
    x, y = split_by_class(x, y, num_classes)
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
    x_labeled = np.vstack(x_labeled)
    y_labeled = np.concatenate(y_labeled)
    x_unlabeled = np.vstack(x_unlabeled)
    y_unlabeled = np.concatenate(y_unlabeled)
    return x_labeled, y_labeled, x_unlabeled, y_unlabeled

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
#x_train, y_train = split_by_class(x_train, y_train, 10)
#x_test, y_test = split_by_class(x_test, y_test, 10)

max_num_labeled = 500
x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled = missing_at_random(x_train, y_train, 10, max_num_labeled)

original_dim = image_size * image_size
x_train_labeled = vectorize_and_normalize(x_train_labeled, original_dim)
x_train_unlabeled = vectorize_and_normalize(x_train_unlabeled, original_dim)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
#    data_test = (np.vstack(x_test), np.concatenate(y_test))
    data_labeled = (x_train_labeled, y_train_labeled)

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

#    if args.weights:
#        vae.load_weights(args.weights)
#    else:
#        # train the autoencoder using both labeled and unlabeled data
#        vae.fit(np.vstack((x_train_labeled, x_train_unlabeled)), epochs=epochs, batch_size=batch_size, validation_data=(np.vstack(x_test), None))
#        vae.save_weights('vae_mlp_mnist.h5')
    vae.load_weights('vae_mlp_mnist.h5')

    # Discrete colormap
    # define the colormap
    cmap = cm.get_cmap("jet", 10)
    bounds = np.linspace(0, 9, 10)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # All data
    z_mean_labeled, _, _ = encoder.predict(x_train_labeled, batch_size=batch_size)
    z_mean_unlabeled, _, _ = encoder.predict(x_train_unlabeled, batch_size=batch_size)
    z_mean_test, _, _ = encoder.predict(x_test, batch_size=batch_size)
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
    plt.scatter(z_mean_labeled[:, 0], z_mean_labeled[:, 1], c=y_train_labeled, cmap=cmap, vmin = -0.5, vmax = 9.5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
#    plt.show()
    plt.title('All training data: labeled and unlabeled')
    plt.savefig("all_data.png")
    
    def plot_nll(z_mean, y, classK, cmap):
        z0 = np.linspace(-6., 6.)
        z1 = np.linspace(-6., 6.)
        Z0, Z1 = np.meshgrid(z0, z1)
        ZZ = np.array([Z0.ravel(), Z1.ravel()]).T
        if classK == -1:
#            z_mean_unlabeled, _, _ = encoder.predict(x_train_unlabeled, batch_size=batch_size)
            gmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full', max_iter=500, tol=1e-3).fit(z_mean)
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
#            plt.figure(figsize=(6, 5))
#            plt.scatter(z_mean_classK[:, 0], z_mean_classK[:, 1], c=cmap(classK/10), cmap=cmap)
#            plt.xlabel("z[0]")
#            plt.ylabel("z[1]")
#            plt.contour(Z0, Z1, nll, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='jet')
#            plt.colorbar()
#            plt.title('Negative log-likelihood: '+str(classK))
#            plt.show()
        
        return gmm
      
    pz_unlabeled = plot_nll(z_mean_unlabeled, None, -1, cmap)
    plt.savefig("gmm_unlabeled.png")
    pz_labeled = plot_nll(z_mean_labeled, y_train_labeled, -2, cmap)
    plt.savefig("gmm_labeled.png")
    pz_labeled_class0 = plot_nll(z_mean_labeled, y_train_labeled, 0, cmap)
#    plt.savefig("gmm_labeled_class0.png")
    pz_labeled_class1 = plot_nll(z_mean_labeled, y_train_labeled, 1, cmap)
#    plt.savefig("gmm_labeled_class1.png")
    pz_labeled_class2 = plot_nll(z_mean_labeled, y_train_labeled, 2, cmap)
#    plt.savefig("gmm_labeled_class2.png")
    pz_labeled_class3 = plot_nll(z_mean_labeled, y_train_labeled, 3, cmap)
#    plt.savefig("gmm_labeled_class3.png")
    pz_labeled_class4 = plot_nll(z_mean_labeled, y_train_labeled, 4, cmap)
#    plt.savefig("gmm_labeled_class4.png")
    pz_labeled_class5 = plot_nll(z_mean_labeled, y_train_labeled, 5, cmap)
#    plt.savefig("gmm_labeled_class5.png")
    pz_labeled_class6 = plot_nll(z_mean_labeled, y_train_labeled, 6, cmap)
#    plt.savefig("gmm_labeled_class6.png")
    pz_labeled_class7 = plot_nll(z_mean_labeled, y_train_labeled, 7, cmap)
#    plt.savefig("gmm_labeled_class7.png")
    pz_labeled_class8 = plot_nll(z_mean_labeled, y_train_labeled, 8, cmap)
#    plt.savefig("gmm_labeled_class8.png")
    pz_labeled_class9 = plot_nll(z_mean_labeled, y_train_labeled, 9, cmap)
#    plt.savefig("gmm_labeled_class9.png")
    
    py_prior = np.ones([10])*.1
    
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
    
    def glrt(pz_labeled, pz_unlabeled, z_mean_unlabeled):
#        z_mean_unlabeled, _, _ = encoder.predict(x_train_unlabeled, batch_size=batch_size)
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
    
#    def assign_labeles_map(lr, z_mean_unlabeled, py_labeled, num_classes):
#        index_similar_features = np.nonzero(lr > 0)
#        post = np.zeros([10, index_similar_features[0].shape[0]])
##        post[0, :] = np.exp(pz_labeled_class0.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[0]
##        post[1, :] = np.exp(pz_labeled_class1.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[1]
##        post[2, :] = np.exp(pz_labeled_class2.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[2]
##        post[3, :] = np.exp(pz_labeled_class3.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[3]
##        post[4, :] = np.exp(pz_labeled_class4.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[4]
##        post[5, :] = np.exp(pz_labeled_class5.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[5]
##        post[6, :] = np.exp(pz_labeled_class6.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[6]
##        post[7, :] = np.exp(pz_labeled_class7.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[7]
##        post[8, :] = np.exp(pz_labeled_class8.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[8]
##        post[9, :] = np.exp(pz_labeled_class9.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[9]
##        
#        post[0, :] = np.exp(pz_labeled_class0.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[0]
#        post[1, :] = np.exp(pz_labeled_class1.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[1]
#        post[2, :] = np.exp(pz_labeled_class2.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[2]
#        post[3, :] = np.exp(pz_labeled_class3.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[3]
#        post[4, :] = np.exp(pz_labeled_class4.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[4]
#        post[5, :] = np.exp(pz_labeled_class5.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[5]
#        post[6, :] = np.exp(pz_labeled_class6.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[6]
#        post[7, :] = np.exp(pz_labeled_class7.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[7]
#        post[8, :] = np.exp(pz_labeled_class8.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[8]
#        post[9, :] = np.exp(pz_labeled_class9.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[9]
#        
#        
#        post_labeles = np.argmax(post, axis=0)
#        return index_similar_features[0], post_labeles
    
    def assign_labeles_map_soft(lr, z_mean_unlabeled, py_labeled, num_classes):
        index_similar_features = np.nonzero(lr > -100)
        index_unfamiliar_features = np.setdiff1d(np.arange(lr.shape[0]), index_similar_features)
        post = np.zeros([10, lr.shape[0]])
#        post[0, :] = np.exp(pz_labeled_class0.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[0]
#        post[1, :] = np.exp(pz_labeled_class1.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[1]
#        post[2, :] = np.exp(pz_labeled_class2.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[2]
#        post[3, :] = np.exp(pz_labeled_class3.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[3]
#        post[4, :] = np.exp(pz_labeled_class4.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[4]
#        post[5, :] = np.exp(pz_labeled_class5.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[5]
#        post[6, :] = np.exp(pz_labeled_class6.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[6]
#        post[7, :] = np.exp(pz_labeled_class7.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[7]
#        post[8, :] = np.exp(pz_labeled_class8.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[8]
#        post[9, :] = np.exp(pz_labeled_class9.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_labeled[9]
        
        post[0, index_similar_features[0]] = np.exp(pz_labeled_class0.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[0]
        post[1, index_similar_features[0]] = np.exp(pz_labeled_class1.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[1]
        post[2, index_similar_features[0]] = np.exp(pz_labeled_class2.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[2]
        post[3, index_similar_features[0]] = np.exp(pz_labeled_class3.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[3]
        post[4, index_similar_features[0]] = np.exp(pz_labeled_class4.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[4]
        post[5, index_similar_features[0]] = np.exp(pz_labeled_class5.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[5]
        post[6, index_similar_features[0]] = np.exp(pz_labeled_class6.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[6]
        post[7, index_similar_features[0]] = np.exp(pz_labeled_class7.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[7]
        post[8, index_similar_features[0]] = np.exp(pz_labeled_class8.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[8]
        post[9, index_similar_features[0]] = np.exp(pz_labeled_class9.score_samples(z_mean_unlabeled[index_similar_features[0]]))*py_prior[9]
        
        post[:, index_unfamiliar_features] = py_prior[0]
        
        post_labeles_soft = np.zeros(lr.shape[0])
        for i in range(lr.shape[0]):
            pvals = post[:, i]/np.sum(post[:, i])
            post_labeles_soft[i] = np.nonzero(np.random.multinomial(1, pvals))[0][0]
        return index_similar_features[0], post_labeles_soft
#    
#    def assign_labeles_mle(lr, z_mean_unlabeled, num_classes):
#        index_similar_features = np.nonzero(lr > 0)
#        ll = np.zeros([10, index_similar_features[0].shape[0]])
#        ll[0, :] = pz_labeled_class0.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[1, :] = pz_labeled_class1.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[2, :] = pz_labeled_class2.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[3, :] = pz_labeled_class3.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[4, :] = pz_labeled_class4.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[5, :] = pz_labeled_class5.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[6, :] = pz_labeled_class6.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[7, :] = pz_labeled_class7.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[8, :] = pz_labeled_class8.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        ll[9, :] = pz_labeled_class9.score_samples(z_mean_unlabeled[index_similar_features[0]])
#        mle_labeles = np.argmax(ll, axis=0)
#        return index_similar_features[0], mle_labeles
    
#    # MLE
#    index_similar_features, mle_labeles = assign_labeles_mle(lr, z_mean_unlabeled, num_classes)
#    index_unfamiliar_features = np.setdiff1d(np.arange(lr.shape[0]), index_similar_features)
#    plt.figure(figsize=(6, 5))
#    plt.scatter(z_mean_unlabeled[index_unfamiliar_features, 0], z_mean_unlabeled[index_unfamiliar_features, 1], c=(0,0,0), alpha=0.05)
#    plt.scatter(z_mean_unlabeled[index_similar_features, 0], z_mean_unlabeled[index_similar_features, 1], c=mle_labeles, cmap=cmap, vmin = -0.5, vmax = 9.5)
#    plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.xlim([-6, 6])
#    plt.ylim([-6, 6])
#    plt.title('MLE of unlabeled training data')
#    plt.savefig("mle_unlabeled.png")
#    
#    index_errors = index_similar_features[np.nonzero(y_train_unlabeled[index_similar_features] != mle_labeles)]
#    plt.figure(figsize=(5, 5))
#    plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
#    plt.scatter(z_mean_unlabeled[index_errors, 0], z_mean_unlabeled[index_errors, 1], c='red', alpha=0.05)
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.xlim([-6, 6])
#    plt.ylim([-6, 6])
#    plt.title('Labeling errors: presentage ' +str(index_errors.shape[0]/index_similar_features.shape[0]))
#    plt.savefig("mle_errors.png")
#    
#    # MAP
#    index_similar_features, map_labeles = assign_labeles_map(lr, z_mean_unlabeled, py_labeled, num_classes)
#    index_unfamiliar_features = np.setdiff1d(np.arange(lr.shape[0]), index_similar_features)
#    plt.figure(figsize=(6, 5))
#    plt.scatter(z_mean_unlabeled[index_unfamiliar_features, 0], z_mean_unlabeled[index_unfamiliar_features, 1], c=(0,0,0), alpha=0.05)
#    plt.scatter(z_mean_unlabeled[index_similar_features, 0], z_mean_unlabeled[index_similar_features, 1], c=map_labeles, cmap=cmap, vmin = -0.5, vmax = 9.5)
#    plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.xlim([-6, 6])
#    plt.ylim([-6, 6])
#    plt.title('MAP of unlabeled training data')
#    plt.savefig("map_unlabeled.png")
#    
#    index_errors = index_similar_features[np.nonzero(y_train_unlabeled[index_similar_features] != map_labeles)]
#    plt.figure(figsize=(5, 5))
#    plt.scatter(z_mean_unlabeled[:, 0], z_mean_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
#    plt.scatter(z_mean_unlabeled[index_errors, 0], z_mean_unlabeled[index_errors, 1], c='red', alpha=0.05)
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.xlim([-6, 6])
#    plt.ylim([-6, 6])
#    plt.title('MAP labeling errors: presentage ' +str(index_errors.shape[0]/index_similar_features.shape[0]))
#    plt.savefig("map_errors.png")
        
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
    
    
    pz_unlabeled_new = plot_nll(z_mean_unlabeled_new, None, -1, cmap)
    plt.savefig("gmm_unlabeled_new.png")
    pz_labeled_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, -2, cmap)
    plt.savefig("gmm_labeled_new.png")
    pz_labeled_class0_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 0, cmap)
#    plt.savefig("gmm_labeled_class0_new.png")
    pz_labeled_class1_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 1, cmap)
#    plt.savefig("gmm_labeled_class1_new.png")
    pz_labeled_class2_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 2, cmap)
#    plt.savefig("gmm_labeled_class2_new.png")
    pz_labeled_class3_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 3, cmap)
#    plt.savefig("gmm_labeled_class3_new.png")
    pz_labeled_class4_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 4, cmap)
#    plt.savefig("gmm_labeled_class4_new.png")
    pz_labeled_class5_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 5, cmap)
#    plt.savefig("gmm_labeled_class5_new.png")
    pz_labeled_class6_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 6, cmap)
#    plt.savefig("gmm_labeled_class6_new.png")
    pz_labeled_class7_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 7, cmap)
#    plt.savefig("gmm_labeled_class7_new.png")
    pz_labeled_class8_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 8, cmap)
#    plt.savefig("gmm_labeled_class8_new.png")
    pz_labeled_class9_new = plot_nll(z_mean_labeled_new, y_train_labeled_new, 9, cmap)
#    plt.savefig("gmm_labeled_class9_new.png")
    
    
    py_labeled_new = np.zeros([10])
    py_labeled_new[0] = np.nonzero(y_train_labeled_new == 0)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[1] = np.nonzero(y_train_labeled_new == 1)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[2] = np.nonzero(y_train_labeled_new == 2)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[3] = np.nonzero(y_train_labeled_new == 3)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[4] = np.nonzero(y_train_labeled_new == 4)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[5] = np.nonzero(y_train_labeled_new == 5)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[6] = np.nonzero(y_train_labeled_new == 6)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[7] = np.nonzero(y_train_labeled_new == 7)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[8] = np.nonzero(y_train_labeled_new == 8)[0].shape[0]/y_train_labeled_new.shape[0]
    py_labeled_new[9] = np.nonzero(y_train_labeled_new == 9)[0].shape[0]/y_train_labeled_new.shape[0]
    
    # For testing data
    post_test = np.zeros([10, y_test.shape[0]])
#    post_test[0, :] = np.exp(pz_labeled_class0_new.score_samples(z_mean_test))*py_labeled_new[0]
#    post_test[1, :] = np.exp(pz_labeled_class1_new.score_samples(z_mean_test))*py_labeled_new[1]
#    post_test[2, :] = np.exp(pz_labeled_class2_new.score_samples(z_mean_test))*py_labeled_new[2]
#    post_test[3, :] = np.exp(pz_labeled_class3_new.score_samples(z_mean_test))*py_labeled_new[3]
#    post_test[4, :] = np.exp(pz_labeled_class4_new.score_samples(z_mean_test))*py_labeled_new[4]
#    post_test[5, :] = np.exp(pz_labeled_class5_new.score_samples(z_mean_test))*py_labeled_new[5]
#    post_test[6, :] = np.exp(pz_labeled_class6_new.score_samples(z_mean_test))*py_labeled_new[6]
#    post_test[7, :] = np.exp(pz_labeled_class7_new.score_samples(z_mean_test))*py_labeled_new[7]
#    post_test[8, :] = np.exp(pz_labeled_class8_new.score_samples(z_mean_test))*py_labeled_new[8]
#    post_test[9, :] = np.exp(pz_labeled_class9_new.score_samples(z_mean_test))*py_labeled_new[9]
    post_test[0, :] = np.exp(pz_labeled_class0_new.score_samples(z_mean_test))*py_prior[0]
    post_test[1, :] = np.exp(pz_labeled_class1_new.score_samples(z_mean_test))*py_prior[1]
    post_test[2, :] = np.exp(pz_labeled_class2_new.score_samples(z_mean_test))*py_prior[2]
    post_test[3, :] = np.exp(pz_labeled_class3_new.score_samples(z_mean_test))*py_prior[3]
    post_test[4, :] = np.exp(pz_labeled_class4_new.score_samples(z_mean_test))*py_prior[4]
    post_test[5, :] = np.exp(pz_labeled_class5_new.score_samples(z_mean_test))*py_prior[5]
    post_test[6, :] = np.exp(pz_labeled_class6_new.score_samples(z_mean_test))*py_prior[6]
    post_test[7, :] = np.exp(pz_labeled_class7_new.score_samples(z_mean_test))*py_prior[7]
    post_test[8, :] = np.exp(pz_labeled_class8_new.score_samples(z_mean_test))*py_prior[8]
    post_test[9, :] = np.exp(pz_labeled_class9_new.score_samples(z_mean_test))*py_prior[9]
    y_test_hat = np.argmax(post_test, axis=0)
    
    pe_test = np.zeros([y_test.shape[0]])
    for i in range(y_test.shape[0]):
        pe_test[i] = 1 - post_test[y_test_hat[i], i]/np.sum(post_test[:, i])
        
    
    index_errors = np.nonzero(y_test != y_test_hat)
    plt.figure(figsize=(5, 5))
    plt.scatter(z_mean_test[:, 0], z_mean_test[:, 1], c=(0,0,0), alpha=0.05)
    plt.scatter(z_mean_test[index_errors, 0], z_mean_test[index_errors, 1], c='red', alpha=0.05)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.title('MAP labeling errors for testing data: presentage ' +str(index_errors[0].shape[0]/y_test.shape[0]))
    plt.savefig("test_map_errors.png")
    
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_test[:, 0], z_mean_test[:, 1], c=pe_test, alpha=0.1, cmap='jet', vmin = 0, vmax = .9)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.title('Error probability')
    plt.savefig("test_pe.png")
    
    z_mean_test_errors = z_mean_test[index_errors, :]
    plt.figure(figsize=(6, 5))
#    plt.hist(pe_test[index_errors[0]])
    plt.scatter(z_mean_test_errors[0, :, 0], z_mean_test_errors[0, :, 1], c=pe_test[index_errors[0]], alpha=0.1, cmap='jet', vmin = 0, vmax = .9)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.title('Error probability of actual errors')
    plt.savefig("test_pe_2.png")
    
    plt.figure(figsize=(6, 5))
    plt.hist(pe_test[index_errors[0]], density=True)
#    plt.scatter(z_mean_test_errors[0, :, 0], z_mean_test_errors[0, :, 1], c=pe_test[index_errors[0]], alpha=0.1, cmap='jet', vmin = 0, vmax = .9)
#    plt.colorbar()
    plt.xlabel("Error probabilities of actual labeling errors")
    plt.ylabel("Density")
#    plt.xlim([-6, 6])
#    plt.ylim([-6, 6])
#    plt.title('Error probability of actual errors')
    plt.savefig("test_pe_3.png")
    
    
    
    