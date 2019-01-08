# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:41:41 2019

@author: Administrator
"""
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Discrete colormap
# define the colormap
cmap = cm.get_cmap("jet", 10)
bounds = np.linspace(0, 9, 10)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

def plot_traindata(z_mean_train, y_train):
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean_train[:, 0], z_mean_train[:, 1], c=y_train, cmap=cmap, vmin = -0.5, vmax = 9.5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
    plt.title('All training data')
    
def plot_digits(decoder):
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
    plt.show()
    
def plot_labeled_unlabeled(z_labeled, y_labeled, z_unlabeled):
    plt.figure(figsize=(6, 5))
    plt.scatter(z_unlabeled[:, 0], z_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
    plt.scatter(z_labeled[:, 0], z_labeled[:, 1], c=y_labeled, cmap=cmap, vmin = -0.5, vmax = 9.5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
    plt.title('Labeled and unlabeled data')
    plt.savefig("train_data.png")
    
def plot_maxlrt(z_unlabeled, lrt_y): 
    plt.figure(figsize=(6, 5))
    plt.scatter(z_unlabeled[:, 0], z_unlabeled[:, 1], c=np.max(lrt_y, axis=1), cmap='jet', vmin = -5, vmax = 5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.colorbar()
    plt.title('LRT for unlabeled data')   
    plt.savefig("lrt.png")
    
def plot_marginallrt(z_unlabeled, lrt_y): 
    plt.figure(figsize=(6, 5))
    plt.scatter(z_unlabeled[:, 0], z_unlabeled[:, 1], c=np.mean(lrt_y, axis=1), cmap='jet', vmin = -5, vmax = 5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.colorbar()
    plt.title('Marginal log-likelihood ratio test for unlabeled data') 
    
def plot_samples(z_unlabeled, samples, y_unlabeled):
    plt.figure(figsize=(6, 5))
    plt.scatter(z_unlabeled[:, 0], z_unlabeled[:, 1], c=samples, cmap=cmap, vmin = -0.5, vmax = 9.5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
    plt.title('Labeled and unlabeled data')
    
    index_errors = np.nonzero(y_unlabeled!= samples)
    plt.figure(figsize=(5, 5))
    plt.scatter(z_unlabeled[:, 0], z_unlabeled[:, 1], c=(0,0,0), alpha=0.05)
    plt.scatter(z_unlabeled[index_errors, 0], z_unlabeled[index_errors, 1], c='red', alpha=0.05)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.title('Posterior sampling errors: presentage ' +str(index_errors[0].shape[0]/y_unlabeled.shape[0]))
    
def plot_test(z_test, y_test_hat, y_test, pe_test):
    index_errors = np.nonzero(y_test != y_test_hat)
#    plt.figure(figsize=(5, 5))
#    plt.scatter(z_test[:, 0], z_test[:, 1], c=(0,0,0), alpha=0.05)
#    plt.scatter(z_test[index_errors, 0], z_test[index_errors, 1], c='red', alpha=0.05)
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.xlim([-6, 6])
#    plt.ylim([-6, 6])
#    plt.title('MAP labeling errors for testing data: presentage ' +str(index_errors[0].shape[0]/y_test.shape[0]))
    
    
    plt.figure(figsize=(6, 5))
    plt.scatter(z_test[:, 0], z_test[:, 1], c=pe_test, alpha=0.1, cmap='jet', vmin = 0, vmax = .9)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.title('Estimated error probability')
    plt.savefig("estimated_pe_2.png")
    
#    z_test_errors = z_test[index_errors, :]
#    plt.figure(figsize=(6, 5))
##    plt.hist(pe_test[index_errors[0]])
#    plt.scatter(z_test_errors[0, :, 0], z_test_errors[0, :, 1], c=pe_test[index_errors[0]], alpha=0.1, cmap='jet', vmin = 0, vmax = .9)
#    plt.colorbar()
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.xlim([-6, 6])
#    plt.ylim([-6, 6])
#    plt.title('Estimated error probability of actual errors')
    
    plt.figure(figsize=(6, 5))
    plt.hist(pe_test[index_errors[0]], density=True)
    plt.xlabel("Estimated error probabilities of \n misclassified testing data")
    plt.ylabel("Density")
    plt.savefig("density_pe_2.png")
    
def plot_topn_pe(x_test, z_test, y_test_hat, y_test, pe_test, decoder, topn):
    n = topn  # Top error probalities
    plt.figure(figsize=(30, 8))
    pe_sorted = pe_test[np.flip(np.argsort(pe_test))]
    
    for i in range(n):
        # display original
        x_decoded = decoder.predict(z_test[np.nonzero(pe_test == pe_sorted[i])[0], :])
#        x_decoded = x_test[np.nonzero(pe_test == pe_sorted[i])[0], :]
        digit = x_decoded[0].reshape(28, 28)
        ax = plt.subplot(1, n, i + 1)
        plt.title('True label: '+str(y_test[np.nonzero(pe_test == pe_sorted[i])[0]])\
                  +'\n predicted label: '+str(y_test_hat[np.nonzero(pe_test == pe_sorted[i])[0]])\
                  +'\n estimated Pe: '+str(np.around(pe_test[np.nonzero(pe_test == pe_sorted[i])[0]], decimals=2)))
        plt.imshow(digit)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
#        x_original = x_test[np.nonzero(pe_test == pe_sorted[i])[0], :]
#        digit = x_original[0].reshape(28, 28)
#        ax = plt.subplot(2, n, i + n + 1)
#        plt.imshow(digit)
#        plt.gray()
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)