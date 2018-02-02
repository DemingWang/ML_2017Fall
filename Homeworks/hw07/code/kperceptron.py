#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 17, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np

## kernels
def linear_kernel(x, z,kparam=None):
    return np.dot(x, z)

def polynomial_kernel(x, z, d):
    return (1 + np.dot(x, z)) ** d

def gaussian_kernel(x, z, sigma):
    return np.exp(-np.linalg.norm(x-z)**2 / (2 * (sigma ** 2)))

def kperceptron_train(X,y,epochs,kernel,kparam):
    # Gram matrix
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples),dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(X[i], X[j],kparam)

    # Get parameter alpha
    alpha = np.zeros(n_samples, dtype=np.float64)
    for t in range(epochs):
        for i in range(n_samples):
            Ki = K[:,i]
            if np.sign(np.sum(Ki * alpha * y)) != y[i]:
                alpha[i] += 1.0
    
    idx = alpha > 1e-5
    alpha = alpha[idx]
    sv = X[idx]
    sv_y = y[idx]
    
    return alpha,sv,sv_y

def kperceptron_test(X,kernel,kparam,alpha,sv, sv_y):
    n_samples, n_features = X.shape

    y_pred = np.zeros(len(X),dtype='f')
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
