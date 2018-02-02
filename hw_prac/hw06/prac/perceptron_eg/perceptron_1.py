#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm.

@author: Bhishan Poudel

@date:  Oct 31, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os, shutil
np.random.seed(100)

def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:,:-1]
    Y = data[:,-1]
    
    return X, Y

def predict(X,w):
    return np.sign(np.dot(X, w))


def perceptron(X, Y,epochs):
    """
    X: data matrix without bias.
    Y: target
    """
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X1 = np.append(ones, X, axis=1)
      
    w = np.zeros(X1.shape[1])
    final_iter = epochs
    
    for epoch in range(epochs):        
        misclassified = 0
        for i, x in enumerate(X1):
            y = Y[i]
            h = np.dot(x, w) * y

            if h <= 0:
                w = w + x*y
                misclassified += 1

        if misclassified == 0:
            final_iter = epoch
            break
        
    print("final_iter = {}".format(final_iter))                
    return w, final_iter

def main():
    """Run main function."""

    X, Y = read_data('data.txt') # X is unbiased
    max_iter = 10000
    w, final_iter = perceptron(X,Y,max_iter)
    print('w = ', w)

if __name__ == "__main__":
    main()
