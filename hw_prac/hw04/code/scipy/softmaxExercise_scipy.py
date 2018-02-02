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

from softmax import softmaxCost, softmaxPredict
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

                                        
    # Fit the model and get theta (theta is flat array)
    theta, _, _ = fmin_l_bfgs_b(softmaxCost, theta,
                                args = (numClasses, inputSize, decay, images, labels),
                                maxiter = 100, disp = 1)

    # # Method 2 Using minimize function from scipy.optimize
    # theta  = scipy.optimize.minimize(softmaxCost, 
    #                                      theta, 
    #                                      args = (numClasses, inputSize, decay, images, labels,), 
    #                                      method = 'L-BFGS-B', 
    #                                      jac = True, 
    #                                      options = {'maxiter': 100}).x

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
    print('Accuracy: %0.3f%%.' % (acc * 100)) # 92.560%.

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
