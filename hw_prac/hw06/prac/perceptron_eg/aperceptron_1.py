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

def aperceptron_sgd(X, Y,epochs):    
    # initialize weights
    w = np.zeros(X.shape[1] )
    u = np.zeros(X.shape[1] )
    b = 0
    beta = 0
    
    # counters    
    final_iter = epochs
    c = 1
    converged = False
    
    # main average perceptron algorithm
    for epoch in range(epochs):
        # initialize misclassified
        misclassified = 0
        
        # go through all training examples
        for  x,y in zip(X,Y):
            h = y * (np.dot(x, w) + b)

            if h <= 0:
                w = w + y*x
                b = b + y
                
                u = u+ y*c*x
                beta = beta + y*c
                misclassified += 1
                
        # update counter regardless of good or bad classification        
        c = c + 1
        
        # break loop if w converges
        if misclassified == 0:
            final_iter = epoch
            converged = True
            # print("Averaged Perceptron converged after: {} iterations".format(final_iter))
            break
    
    # if converged == False:
    #     print("Averaged Perceptron DID NOT converged.")
    # 
    # return w and final_iter
    w = w - u/c
    b = np.array([b- beta/c])
    w = np.append(b, w)
    print("final_iter = {}".format(final_iter))
    return w, final_iter

def main():
    """Run main function."""

    X, Y = read_data('data.txt') # X is without bias
    max_iter = 20
    w, final_iter = aperceptron_sgd(X,Y,max_iter)
    print('w = ', w)
    
if __name__ == "__main__":
    main()
