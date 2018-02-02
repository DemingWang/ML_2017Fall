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

def kperceptron_train(data,labels,epochs,kernel,kparam):
    X = data
    y = labels
    y = np.array(y)
    y[y==0] = -1
    
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
            
    return np.sign(y_pred), y_pred


##=========================Read data===================================
def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:,:-1]
    Y = data[:,-1]
       
    return X, Y

def main():
    """Run main function."""
    # data file
    data_tr = '../data/extra/train.txt'
    data_ts = '../data/extra/test.txt'
    
    ## data   
    X_train, y_train = read_data(data_tr)
    X_test,  y_test = read_data(data_ts)
    
    # kernels
    # epochs,kernel, kparam = 200, gaussian_kernel, 0.5
    epochs,kernel, kparam = 200, polynomial_kernel, 3
    
    # fit the kernel perceptron
    alpha, sv, sv_y = kperceptron_train(X_train,y_train,epochs,kernel,kparam)
    
    y_pred, hyp = kperceptron_test(X_test,kernel,kparam,alpha,sv,sv_y)
    
    
    # correct
    correct = np.sum(y_pred == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_pred)))

if __name__ == "__main__":
    main()
