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

def loadMNISTImages(file_name):

    # Read binary file
    image_file = open(file_name, 'rb')
    
    # Read header information from the file    
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)    
    # print("head1 = {}".format(head1)) # b'\x00\x00\x08\x03'
    
    
    # Format the header information for useful data    
    num_examples = struct.unpack('>I', head2)[0]
    num_rows     = struct.unpack('>I', head3)[0]
    num_cols     = struct.unpack('>I', head4)[0]
    # print("num_examples = {}".format(num_examples)) # 60000
    # print("num_rows = {}".format(num_rows)) # 28
    # print("num_cols = {}".format(num_cols)) # 28
    
    # Initialize dataset as array of zeros  
    dataset = np.zeros((num_rows*num_cols, num_examples))
    # print("dataset.shape = {}".format(dataset.shape)) # (784, 60000)
    # print("type(dataset) = {}".format(type(dataset)))       # <class 'np.ndarray'>
    
    # Read the actual image data   
    images_raw  = array.array('B', image_file.read())
    image_file.close()
    # print("images_raw[0:5] = {}".format(images_raw[0:5])) # array('B', [0, 0, 0, 0, 0])
    
    # Arrange the data in columns
    # Get np array of dataset from array.array of images_raw  
    for i in range(num_examples):
    
        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)
        
        dataset[:, i] = images_raw[limit1 : limit2]
    
    # Normalize and return the dataset
    dataset = dataset / 255
    # print("type(images_raw) = {}".format(type(images_raw))) # <class 'array.array'>
    # print("type(dataset) = {}".format(type(dataset)))       # <class 'np.ndarray'>
            
    return dataset

    
def loadMNISTLabels(file_name):

    # Read binary fle
    label_file = open(file_name, 'rb')
    
    # Read headers
    head1 = label_file.read(4)
    head2 = label_file.read(4)    
    num_examples = struct.unpack('>I', head2)[0]
    
    # Initialize empty np array for labels 
    labels = np.zeros((num_examples, 1), dtype = np.int)
    
    # Read binary file of type array.array  
    labels_raw = array.array('b', label_file.read())
    label_file.close()
    
    # Get np array from array.array datatype 
    labels[:, 0] = labels_raw[:]
    
    return labels

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
    pred = np.argmax(theta_x, axis=1)[None].T
    
    
    # shapes
    # print("theta.shape after = {}".format(theta.shape)) # (10, 784)
    # print("data.shape = {}".format(data.shape))  # (784, 10000)
    # print("pred.shape = {}".format(pred.shape))  # (10000, 1)
    
    return pred


def runSoftmaxRegression():
    
    # Initialize parameters of the Regressor   
    input_size     = 784    # input vector size
    num_classes    = 10     # number of classes
    decay          = 0.0001 # weight decay parameter
    max_iterations = 100    # number of optimization iterations
    
    # Train data
    training_data   = loadMNISTImages('data/train-images-idx3-ubyte')
    training_labels = loadMNISTLabels('data/train-labels-idx1-ubyte')
    
    X = training_data
    data = training_data
    labels = training_labels
    
    # print("X.shape = {}".format(X.shape)) # (784, 60000)
    # print("labels.shape = {}".format(labels.shape)) # (60000, 1)
    
    # Get ground_truth
    ground_truth = getGroundTruth(labels)
    # print("ground_truth.shape = {}".format(ground_truth.shape)) # (10, 60000)
    
    # Get cost and gradient
    np.random.seed(100)
    theta = 0.005 * np.asarray(np.random.normal(size = (num_classes*input_size, 1)))
    cost, grad = softmaxCost(theta, data, labels, decay, input_size, num_classes)
    # print("cost = {}".format(cost)) # 2.2988103841518273
    # print("grad.shape = {}".format(grad.shape)) # (1, 7840)
    
    # fit the model
    # opt_theta, _, _ = fmin_l_bfgs_b(softmaxCost, theta,
    #                             args = (data, labels, decay, input_size, num_classes),
    #                             maxiter = 100, disp = 1)

    opt_theta  = scipy.optimize.minimize(softmaxCost, 
                                         theta, 
                                         args = (data, labels, decay, input_size, num_classes,), 
                                         method = 'L-BFGS-B', 
                                         jac = True, 
                                         options = {'maxiter': max_iterations}).x
    
    # print("theta.shape = {}".format(theta.shape))
    print("\n\n")
    
    # Test
    test_data   = loadMNISTImages('data/t10k-images-idx3-ubyte') 
    test_labels = loadMNISTLabels('data/t10k-labels-idx1-ubyte')
    
    
    pred = softmaxPredict(opt_theta, test_data, num_classes, input_size)
    
    
    correct = test_labels[:, 0] == pred[:, 0]
    print("""\n\nAccuracy : """, np.mean(correct*100), "%")
    
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
