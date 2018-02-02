#!python
# -*- coding: utf-8 -*-#
"""
Softmax Regression Using Scipy.

@author: Bhishan Poudel

@date:  Oct 16, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import argparse
import sys
import numpy as np
from numpy.random import randn, randint
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b
import scipy
from tqdm import tqdm

from softmax import softmaxCost, softmaxPredict, softmaxGrad
from computeNumericalGradient import computeNumericalGradient
from checkNumericalGradient import checkNumericalGradient

def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser('Softmax Exercise.')

    # Add a argument
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../../data/mnist/',
                        help='Directory to put the input MNIST data.')

    # Add another argument
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Used for gradient checking.')

    FLAGS, unparsed = parser.parse_known_args()

    print("FLAGS.input_data_dir = {}".format(FLAGS.input_data_dir))
    print("FLAGS.debug = {}".format(FLAGS.debug))

    return FLAGS

def shuffle_and_split(X, y, chunk_size):
    """Shuffle and split the given data and labels.
    
    Parameters
    -------------
    
    X: input matrix of shape N * M (with N samples and M features)
    
    y: Target (1d array)
    
    chunk_size: size of each chunks in the splits 
         
     """
    
    
    # X is design matrix (rows are examples), t is 1d array
    X = np.array(X).T
    y = np.array(y)
    y = y.reshape(len(y),1)
    data = np.append(X,y,axis=1)
    num_splits = int(X.shape[0]/chunk_size)
    
    # debug
    # print("\n\nInside shuffle")
    # print("batchsize = {}".format(batchsize))
    # print("num_splits = {}".format(num_splits))
    # print("images.shape = {}".format(images.shape)) # (784, 55000)
    # print("labels.shape = {}".format(labels.shape)) # (55000,)
    
    
    # shuffle data
    np.random.seed(100)
    perm_idx = np.random.permutation(X.shape[0])
    X2 = X[perm_idx]
    y2 = y[perm_idx]
    
    # After shuffling split the data
    data2 = np.append(X2,y2,axis=1)    
    batches = np.array_split(data2, num_splits)
    
    # debug
    # print("len(batches) = {}".format(len(batches)))
    # print("data2.shape = {}".format(data2.shape))
    
    # Again split into X, and y
    # X = chunks[0][:,:-1]  # 2d array with each row is an example
    # y = chunks[0][:,-1]   # 1d array (not column vector [:,-1:])
    # print("X = {}".format(X))
    # print("y = {}".format(y))
    
    return batches
 
def minibatch_grad_desc(theta_init, max_iters,batches,numClasses,inputSize,decay,learning_rate,batchsize):
    print("Fitting the params using minibatch gradient descent model with learning_rate = {} ...\n\n".format(learning_rate))
    cost_lst = []
    for n in tqdm(range(max_iters)):
        for i, batch in enumerate(batches):
            img, lbl = batches[i][:,:-1], batches[i][:,-1]
            # print("img.shape = {}".format(img.shape)) # (100, 784)
            # print("lbl.shape = {}".format(lbl.shape)) # (100, )
            # gradient = softmaxGrad(theta_init, numClasses, inputSize, decay, img.T, lbl)   
            cost, gradient = softmaxCost(theta_init, numClasses, inputSize, decay, img.T, lbl)   
            theta = theta - learning_rate/batchsize * gradient
            
            # Add costs to plot cost versus, iterations
            cost_lst.append(cost)
                
            
    return theta, cost_lst

def split_train_valid():
    # Load training data
    images = np.load('../../data/mnist/train-images_55k.npy')
    labels = np.load('../../data/mnist/train-labels_55k.npy')
    
    # reshape
    images = images.reshape(55000,784)
    labels = labels.reshape(55000,1)
    
    train_img, train_lbl = images[0:50000, :], labels[0:50000, :]
    valid_img, valid_lbl = images[50000:, :], labels[50000:, :]
    
    # Now save train and valid images and labels
    np.savetxt('../../data/mnist/train-images.csv', train_img)
    np.savetxt('../../data/mnist/train-labels.csv', train_lbl)

    np.savetxt('../../data/mnist/valid-images.csv', valid_img)
    np.savetxt('../../data/mnist/valid-labels.csv', valid_lbl)

    print("\n\n For MNIST train data")
    print("images.shape = {}".format(images.shape)) # (784, 55000)
    print("labels.shape = {}".format(labels.shape)) # (55000,1)
    
    print("\nimages[0].shape = {}".format(images[0].shape)) # (784,)
    print("labels[0].shape   = {}".format(labels[0].shape)) # (1,)
    
    print("\nlen(images[0]) = {}".format(len(images[0]))) # 784
    print("len(labels[0])   = {}".format(len(labels[0]))) # 1
    
    print("\ntrain_img.shape = {}".format(train_img.shape)) # 
    print("train_lbl.shape   = {}".format(train_lbl.shape)) # 
    
    print("\nvalid_img.shape = {}".format(valid_img.shape)) # 
    print("valid_lbl.shape   = {}".format(valid_lbl.shape)) # 
    print("\n\n")

    
def softmax_scipy():

    FLAGS = parse_args()

    # Initiliaze values
    inputSize = 28 * 28 # Size of input vector (MNIST images are 28x28)
    numClasses = 10     # Number of classes (MNIST images fall into 10 classes)
    decay = 1e-4        # Weight decay parameter

    # Load training data
    images = np.load(FLAGS.input_data_dir + 'train-images.npy')
    labels = np.load(FLAGS.input_data_dir + 'train-labels.npy')
    print("\n\n For MNIST train data")
    print("images.shape = {}".format(images.shape)) # (784, 55000)
    print("labels.shape = {}".format(labels.shape)) # (55000,)
    print("\n\n")

    # -------------------------------------------------------
    # Create data for debugging
    if FLAGS.debug:
        inputSize = 8
        np.random.seed(100)
        images = randn(8, 100)
        labels = randint(0, 10, 100, dtype = np.uint8)

    # Randomly initialise theta (theta is 1d array)
    np.random.seed(100)
    theta_init = 0.005 * randn(numClasses * inputSize)

    # Get cost and grad
    cost, grad = softmaxCost(theta, numClasses, inputSize, decay, images, labels)


    # ---------------- debug: Gradient Checking Start ------------------------
    if FLAGS.debug:
        checkNumericalGradient()

        numGrad = computeNumericalGradient(
                    lambda x: softmaxCost(x, numClasses, inputSize, decay, images, labels),
                    theta
                    )

        # Use this to visually compare the gradients side by side.
        print(np.stack((numGrad, grad)).T)


        # Compare numerically computed gradients with those computed analytically.
        diff = norm(numGrad - grad) / norm(numGrad + grad)
        print(diff)
        sys.exit(1)
    # ---------------- debug: Gradient Checking End ------------------------
    max_iters     = 2000
    learning_rate = 0.1
    batchsize     = 100
    batches       = shuffle_and_split(images, labels, batchsize)
    # print("batches[0].shape = {}".format(batches[0].shape)) # (100, 785)
    
    # Fit the parameter theta
    theta, cost_lst = minibatch_grad_desc(theta_init, max_iters,batches,numClasses,inputSize,decay,learning_rate,batchsize)
    
    # Test the data
    images = np.load(FLAGS.input_data_dir + 'test-images.npy')
    labels = np.load(FLAGS.input_data_dir + 'test-labels.npy')
    print("\n\n For MNIST test data")
    print("images.shape = {}".format(images.shape)) # (784, 10000)
    print("labels.shape = {}".format(labels.shape)) # (10000,)
    print("\n\n")

    # Get prediction for test data
    theta = np.reshape(theta, (numClasses, inputSize))
    pred = softmaxPredict(theta, images)
    acc = np.mean(labels == pred)
    print('Accuracy: %0.3f%%.' % (acc * 100)) # 92.630%. (for eta = 10)

def plot_cost_epoch(Jvals_lst, epochs):
    # matplotlib customization
    plt.style.use('ggplot')

    # without lr 1 and 10
    plt.plot(epochs, Jvals_lst[0], label='learning rate = 0.0001')
    plt.plot(epochs, Jvals_lst[1], label='learning rate = 0.001')
    plt.plot(epochs, Jvals_lst[2], label='learning rate = 0.01')
    plt.plot(epochs, Jvals_lst[3], label='learning rate = 0.1')
    plt.xlabel('epoch')
    plt.ylabel('Cost  J(w)')
    plt.title('Choosing hyperparameter learning_rate')
    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/cost_epochs_good_lr.png')
    plt.show()
    plt.close()

    # For learning rate 1 and 10 we get nans
    plt.plot(epochs, Jvals_lst[4], label='learning rate = 1')
    plt.plot(epochs, Jvals_lst[5], label='learning rate = 10')
    plt.xlabel('epoch')
    plt.ylabel('Cost  J(w)')
    plt.title('Choosing hyperparameter learning_rate')
    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/cost_epochs_bad_lr.png')
    plt.show()
    plt.close()

    
def choose_learning_rate():
    max_iters,numClasses,inputSize,decay,batchsize = 2000, 10, 5000, 1e-4 , 100
    learning_rates = [10**i for i in range(-4,2)]
    
    # Load training data
    images  = np.genfromtxt('../../data/mnist/valid-images.csv',delimiter=None,dtype=np.float64) # (5000, 784)
    labels  = np.genfromtxt('../../data/mnist/valid-labels.csv',delimiter=None,dtype=np.float64) #  (5000,)
    
    print("images.shape = {}".format(images.shape))
    print("lables.shape = {}".format(labels.shape))
    batches = shuffle_and_split(images.T, labels, batchsize)
    
    theta_init = 0.005 * randn(numClasses * inputSize)
    for learning_rate in learning_rates:
        _, cost =  minibatch_grad_desc(theta_init, max_iters,batches,numClasses,inputSize,decay,learning_rate,batchsize)
    print("len(cost) = {}".format(len(cost)))
                    
    
    # plot_cost_epoch(Jvals_lst, range(len(Jvals_lst[0])))


def main():
    """Run main function."""
    softmax_scipy()

if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    #  Run the main program
    # main()
    # split_train_valid()
    choose_learning_rate()

    # Print the time taken
    program_end_time = time.time()
    end_ctime        = time.ctime()
    seconds          = program_end_time - program_begin_time
    m, s             = divmod(seconds, 60)
    h, m             = divmod(m, 60)
    d, h             = divmod(h, 24)
    print("\nBegin time: ", begin_ctime)
    print("End   time: ", end_ctime, "\n")
    print("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
