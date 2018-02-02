#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 7, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np

## kernels
def linear_kernel(x1, x2,kparam=None):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, d):
    return (1 + np.dot(x, y)) ** d

def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def kperceptron_train(X,y,epochs,kernel,kparam):    
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples, dtype=np.float64)

    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(X[i], X[j],kparam)

    for t in range(epochs):
        for i in range(n_samples):
            if np.sign(np.sum(K[:,i] * alpha * y)) != y[i]:
                alpha[i] += 1.0
    
    idx = alpha > 1e-5
    alpha = alpha[idx]
    sv = X[idx]
    sv_y = y[idx]
    
    return alpha,sv,sv_y

def kperceptron_test(X,kernel,kparam,alpha,sv, sv_y):
    n_samples, n_features = X.shape

    y_pred = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, svy, sv_ in zip(alpha, sv_y, sv):
            s += a * svy * 1
            s += a * svy * kernel(X[i], sv_,kparam)
            y_pred[i] = s
            
    y_pred_sign = np.sign(y_pred)            
    return y_pred_sign, y_pred


##=========================Read data===================================
def read_data(infile):
    data = np.genfromtxt(infile,delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    
    X = np.array(X).astype(np.float64)
    y = np.array(y).astype(np.float64)
       
    return X, y
