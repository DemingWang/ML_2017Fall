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
import matplotlib.pyplot as plt
np.random.seed(100)


##=======================================================================
## Kernel Functions
##=======================================================================

def linear_kernel(x1, x2,kparam=1):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, p):
    return (1 + np.dot(x1, x2)) ** p

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))

def kperceptron_train(Xtrain,ytrain,epochs,kernel,kparam):
    # initialize alpha
    n_samples = Xtrain.shape[0]
    alpha = np.zeros(n_samples, dtype=np.float64)

    # gram marix
    K = np.zeros((n_samples, n_samples),dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(Xtrain[i], Xtrain[j],kparam)
            
    # fit the model
    for t in range(epochs):
        for i in range(n_samples):
            if np.sign(np.sum(K[:,i] * alpha * ytrain)) != ytrain[i]:
                alpha[i] += 1.0
    
    # support vectors
    idx = alpha > 1e-5
    alpha = alpha[idx]
    sv = Xtrain[idx]
    sv_y = ytrain[idx]
    
    return alpha,sv,sv_y

def kperceptron_test(Xdevtest,kernel,alpha,sv,sv_y,kparam):
    # initialize y_pred
    y_pred = np.zeros(len(Xdevtest),dtype=np.float64)
    
    # predict y
    for i in range(len(Xdevtest)):
        s = 0
        for a, svy, sv_ in zip(alpha, sv_y, sv):
            s += a * svy * 1
            s += a * svy * kernel(Xdevtest[i], sv_,kparam)
        y_pred[i] = s
        
    y_pred_sign = np.sign(y_pred)
    return y_pred, y_pred_sign


##=========================Read data===================================
def read_data(infile):
    data = np.genfromtxt(infile,delimiter=',')
    X = np.array(data[:,:-1]).astype(np.float64) # n_samples, n_features
    y = np.array(data[:,-1]).astype(np.float64)  # n_samples,   # 1d array       
    return X, y
