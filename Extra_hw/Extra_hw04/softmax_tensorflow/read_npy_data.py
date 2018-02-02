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
import numpy as np
import matplotlib.pyplot as plt

def read_npy_data():
        
    # Train data
    tr_data = np.load('data/train-image-py.npy')
    tr_labels = np.load('data/trainlabel-py.npy')
    
    ts_data = np.load('data/t10k-iimage-py.npy')
    ts_labels = np.load('data/t10k-label-py.npy')
    
    print(("tr_data.shape   = {}".format(tr_data.shape)))   # (60000, 28, 28)
    print(("tr_labels.shape = {}".format(tr_labels.shape))) # (60000,)
    
    print(("ts_data.shape   = {}".format(ts_data.shape)))   # (10000, 28, 28)
    print(("ts_labels.shape = {}".format(ts_labels.shape))) # (10000,)
   
def plot_data():
    train_image = np.load('data/train-image-py.npy')
    train_label = np.load('data/trainlabel-py.npy')
    test_image = np.load('data/t10k-iimage-py.npy')
    test_label = np.load('data/t10k-label-py.npy')
    
    im = train_image[9,:,:]
    im = 255*im
    plt.imshow(im, cmap='gray')
    print((train_label[9]))
    # plt.show()
    
    im = test_image[17,:,:]
    im = 255*im
    plt.imshow(im, cmap='gray')
    print((test_label[17]))
    plt.show()
    
def reformat_data():
    train_image = np.load('data/train-image-py.npy')
    train_label = np.load('data/trainlabel-py.npy')
    test_image = np.load('data/t10k-iimage-py.npy')
    test_label = np.load('data/t10k-label-py.npy')
    
    image_size = 28
    num_labels = 10
    num_channels = 1 # gray scale

    reformat = lambda data,labels: (data.reshape((-1, image_size, image_size, 1)).astype(np.float32),(np.arange(num_labels) == labels[:,None]).astype(np.float32))
    
    train_dataset, train_labels = reformat(train_image, train_label)
    test_dataset, test_labels = reformat(test_image, test_label)
    
    print(('train_dataset size: ', train_dataset.shape)) # (60000, 28, 28, 1)
    print(('train_labels size: ', train_labels.shape))   # (60000, 10)
    print(('test_dataset size: ', test_dataset.shape))   # (10000, 28, 28, 1)
    print(('test_labels size: ', test_labels.shape))     # (10000, 10)
    
    
    accuracy = lambda pred, labels: (100.0 * np.sum(np.argmax(pred,1) == np.argmax(labels,1))/pred.shape[0] )
  
    
def main():
    # read_npy_data()
    # plot_data()
    reformat_data()
    
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
    print(("\n\nBegin time: ", begin_ctime))
    print(("End   time: ", end_ctime, "\n"))
    print(("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s)))
