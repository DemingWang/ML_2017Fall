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

def polynomial_kernel(x, y, d=1):
    return (1 + np.dot(x, y)) ** d

def gaussian_kernel(x, y, sigma=1):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def Gram_matrix(X1,kernel,kparam):    
    n_samples = X1.shape[0]
    
    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(X1[i], X1[j],kparam)
            
#    print("gram matrix shape is {}".format(K.shape)) # n_samples,n_samples
    return K

def kperceptron_train(X1,y,epochs,K):
    n_samples = X1.shape[0]
    alpha = np.zeros(n_samples, dtype=np.float64)

    for t in range(epochs):
        for i in range(n_samples):
            Ki = K[:,i]
            if np.sign(np.sum(Ki * alpha * y)) != y[i]:
                alpha[i] += 1.0
    
    idx = alpha > 1e-5
    alpha = alpha[idx]
    sv = X1[idx]
    sv_y = y[idx]
    
    return alpha,sv,sv_y

def kperceptron_test(X1,kernel,kparam,alpha,sv, sv_y):
    n_samples = X1.shape[0]

    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        s = 0
        for a, svy, sv_ in zip(alpha, sv_y, sv):
            s += a * svy * 1
            s += a * svy * kernel(X1[i], sv_,kparam)
            y_pred[i] = s
            
    y_pred_sign = np.sign(y_pred)            
    return y_pred_sign, y_pred


##=========================Read data===================================
def read_data(infile):
    
    data = np.genfromtxt(infile,dtype='f')
    X = data[:,:-1]
    y = data[:,-1]
    
    X1 = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    y = np.array(y)
    

    return X, X1, y


def main():
    """Run main function."""
    X, X1, y = read_data('../data/separable/train.txt')
    Xt, Xt1, yt = read_data('../data/separable/test.txt')
    print('x1 shape', X1.shape) # 60,3

    
    epochs = 4
#    kernel,kparam = gaussian_kernel, 10
    kernel,kparam = polynomial_kernel, 2
    
    K = Gram_matrix(X1,kernel,kparam)
    alpha,sv,sv_y = kperceptron_train(X1,y,epochs,K)
    
    y_pred_sign, y_pred = kperceptron_test(Xt1,kernel,kparam,alpha,sv, sv_y)
    correct = sum(y_pred_sign == yt)
    print("correct = {} out of {}".format(correct, len(y_pred)))
    

if __name__ == "__main__":
    main()
