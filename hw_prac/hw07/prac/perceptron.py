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

def perceptron(X, Y,epochs):
        
    w = np.zeros(X.shape[1])
    final_iter = epochs
    
    for epoch in range(epochs):        
        
        misclassified = 0
        for i, x in enumerate(X):
            y = Y[i]
            h = np.dot(x, w)*y

            if h <= 0:
                w = w + x*y
                misclassified += 1
                
        if misclassified == 0:
            final_iter = epoch
            break
                
    return w
