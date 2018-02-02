#!python
# -*- coding: utf-8 -*-#
"""
Softmax Regression for MNIST data.

@author: Bhishan Poudel

@date: Oct 15, 2017

@email: bhishanpdl@gmail.com

Ref: https://ludlows.github.io/2016-08-11-Recognition-MNIST-Handwriting-Digits/
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


def main():
        
    # Train data
    tr_data   = loadMNISTImages('data/train-images-idx3-ubyte')
    tr_labels = loadMNISTLabels('data/train-labels-idx1-ubyte')
        
    # Test
    ts_data   = loadMNISTImages('data/t10k-images-idx3-ubyte') 
    ts_labels = loadMNISTLabels('data/t10k-labels-idx1-ubyte')
    
    print("tr_data.shape   = {}".format(tr_data.shape))   # (784, 60000)
    print("tr_labels.shape = {}".format(tr_labels.shape)) # (60000, 1)
    
    print("ts_data.shape   = {}".format(ts_data.shape))   # (784, 10000)
    print("ts_labels.shape = {}".format(ts_labels.shape)) # (10000, 1)
   

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
    print("\n\nBegin time: ", begin_ctime)
    print("End   time: ", end_ctime, "\n")
    print("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
