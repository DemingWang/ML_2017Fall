#!python
# -*- coding: utf-8 -*-#
"""
Softmax Regression for MNIST data.

@author: Bhishan Poudel

@date: Oct 15, 2017

@email: bhishanpdl@gmail.com

Ref: http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression
"""
# Imports
import struct
import array
import time
import scipy.sparse
import scipy.optimize
import numpy as np
from scipy.optimize import fmin_l_bfgs_b # This prints too many lines

def getGroundTruth(labels):

    labels = np.array(labels).flatten()
    
    data   = np.ones(len(labels))
    indptr = np.arange(len(labels)+1)
    
    ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
    ground_truth = np.transpose(ground_truth.todense())
    
    return ground_truth

    
def softmaxCost(theta, data, labels, decay, input_size, num_classes):
    """Compute cost and gradient.
    
    
    Args:
    
        theta : 1d array of len n_labels * n_unique_lables e.g (7840,)
                we reshpae theta to  n_unique_lables, n_labels e.g 10, 784
        
        data : shape of data is all_labels, all_examples
               e.g. shape is (784, 60000) 
        
        labels: labels of each examples.
                e.g 60,000 examples in MNIST data has 60k labels.
                from labels we create ground_truth matrix.
                shape of labels = (60000, 1)
                
        decay: decay or lambda parameter of regularized cost function. e.g 0.0001
        
        input_size: size of one example e.g. 784
        
        num_classes: Number of unique classes e.g 10


        
    Return:
    
        cost (float): cost
        theta_grad : gradient        
            
            
    """
    
    # shapes
    # print("theta.shape = {}".format(theta.shape)) # (7840,)  we make it 10, 784
    # print("data.shape = {}".format(data.shape)) # (784, 60000)
    # print("labels.shape = {}".format(labels.shape)) # (60000, 1)

    N = data.shape[1]
    X = data
    
    delta = getGroundTruth(labels)
    theta = theta.reshape(num_classes, input_size)
    z     = theta @ data

    hyp  = np.exp(z-np.amax(z, axis=0, keepdims=True))       
    prob = hyp / np.sum(hyp, axis = 0)
    
    cost = np.multiply(delta, np.log(prob)) # * operator fails here.
    cost = -(np.sum(cost) / N)
    
    weight_decay  = decay/2 * np.sum(theta**2)        
    cost = cost + weight_decay
    
    theta_grad = -1/N * (delta - prob) @  X.T + decay * theta
    theta_grad = theta_grad.flatten()
    
    
    # print("\n\n")
    # print("theta.shape = {}".format(theta.shape)) # (7840,)  we make it 10, 784
    # print("data.shape = {}".format(data.shape)) # (784, 60000)
    # print("labels.shape = {}".format(labels.shape)) # (60000, 1)
    # print("theta.shape = {}".format(theta.shape)) # (7840,)  we make it 10, 784
    # print("ground_truth.shape = {}".format(delta.shape)) # 10, 60,000
    # print("\n\n")
    return [cost, theta_grad]

        
def softmaxPredict(theta, data, num_classes, input_size):
    """Predict the labels of each samples in the data.
    
    Args:
    
        theta : 1d array of len coluns in data (data.shape[1] is num_samples.) 
                e.g. (7840,)
                
        data  : matrix with ncolumns as number of samples. 
                e.g. shape = (784, 10000)
                
        num_classes: Number of unique classes e.g 10    

        input_size: size of one example e.g. 784
    
    Return:
        pred: column vector of predictions to each examples.
              e.g. for 10k test examples shape is (10000, 1)
            
    """
        
    # print("theta.shape before = {}".format(theta.shape)) # (7840,)
    theta    = theta.reshape(num_classes, input_size)        
    theta_x  = data.T @ theta.T  # theta_x can be called z
    pred = np.argmax(theta_x, axis=1)
    
    
    # shapes
    print("theta.shape after = {}".format(theta.shape)) # (10, 784)
    print("data.shape = {}".format(data.shape))  # (784, 10000)
    print("pred.shape = {}".format(pred.shape))  # (10000, )
    
    return pred


def runSoftmaxRegression():
    
    # Initialize parameters of the Regressor   
    input_size     = 784    # input vector size 28*28 = 784
    num_classes    = 10     # number of classes 0 to 9
    decay          = 0.0001 # weight decay parameter
    max_iterations = 100    # number of optimization iterations
    
    # Train data
    training_data   = np.load('data/train-images.npy')
    training_labels = np.load('data/train-labels.npy')
    
    X = training_data
    data = training_data
    labels = training_labels
    
    print("X.shape = {}".format(X.shape)) # (784, 55000)
    print("labels.shape = {}".format(labels.shape)) #   (55000,)
    
    # Get ground_truth
    ground_truth = getGroundTruth(labels)
    print("ground_truth.shape = {}".format(ground_truth.shape)) # (10, 55000)
    
    # Get cost and gradient
    np.random.seed(100)
    theta = 0.005 * np.asarray(np.random.normal(size = (num_classes*input_size, 1)))
    cost, grad = softmaxCost(theta, data, labels, decay, input_size, num_classes)
    print("cost = {}".format(cost)) # 2.29
    print("grad.shape = {}".format(grad.shape)) # (1, 7840)
    
    # fit the model
    # Fitting using scipy fmin (This prints too many things)
    # opt_theta, _, _ = fmin_l_bfgs_b(softmaxCost, theta,
    #                             args = (data, labels, decay, input_size, num_classes),
    #                             maxiter = 100, disp = 1)

    # Fitting using scipy optimize minimize.
    opt_theta  = scipy.optimize.minimize(softmaxCost, 
                                         theta, 
                                         args = (data, labels, decay, input_size, num_classes,), 
                                         method = 'L-BFGS-B', 
                                         jac = True, 
                                         options = {'maxiter': max_iterations}).x
    
    print("opt_theta.shape = {}".format(opt_theta.shape)) # (7840, 1)
    print("\n\n")
    
    # Test
    test_data   = np.load('data/test-images.npy') 
    test_labels = np.load('data/test-labels.npy')
    print("test_data.shape = {}".format(test_data.shape))     # (784, 10000)
    print("test_labels.shape = {}".format(test_labels.shape)) # (10000,)
    
    
    pred = softmaxPredict(opt_theta, test_data, num_classes, input_size)
    print("pred.shape = {}".format(pred.shape)) # (10000, 1)
    
    
    acc = np.mean(test_labels == pred)
    print('Accuracy: %0.3f%%.' % (acc * 100))
    
def main():
    """Run main function."""
    runSoftmaxRegression()
    

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
