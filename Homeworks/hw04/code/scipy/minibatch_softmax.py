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
    theta = 0.005 * randn(numClasses * inputSize)

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
    max_iters = 2000
    eta       = 0.1
    batchsize = 100
    batches   = shuffle_and_split(images, labels, batchsize)
    # print("batches[0].shape = {}".format(batches[0].shape)) # (100, 785)
    
    print("Fitting the params using minibatch gradient descent model with eta = {} ...\n\n".format(eta))   
    for n in tqdm(range(max_iters)):
        for i, batch in enumerate(batches):
            img, lbl = batches[i][:,:-1], batches[i][:,-1]
            # print("img.shape = {}".format(img.shape)) # (100, 784)
            # print("lbl.shape = {}".format(lbl.shape)) # (100, )
            gradient = softmaxGrad(theta, numClasses, inputSize, decay, img.T, lbl)   
            theta = theta - eta/batchsize * gradient
    
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

def main():
    """Run main function."""
    softmax_scipy()

if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    #  Run the main program
    main()

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
