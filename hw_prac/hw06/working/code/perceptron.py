#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 7, 2017

"""
# Imports
import collections
import numpy as np
import itertools

##===================Vanilla Perceptron================================
def perceptron_train(data, labels, epochs,verbose=False):
    X = data
    Y = labels
    # print(Y[0:10])
    
    # change labels 0 to -1
    Y[Y==0] = -1
    
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X = np.append(ones, X, axis=1)
    
    w = np.zeros(X.shape[1])
    final_iter = epochs
    mistakes = 0
    
    # debug
    # print("X.shape = {}".format(X.shape)) # examples, features+1
    # print("w.shape = {}".format(w.shape)) # features+1,
    # print("Y.shape = {}".format(Y.shape)) # examples,

    
    for epoch in range(epochs):
        if verbose:
            print("\n")
            print("epoch: {} {}".format(epoch, '-'*40))
        
        
        misclassified = 0
        for i, x in enumerate(X):
            y = Y[i]
            h = np.dot(x, w)*y

            if h <= 0:
                w = w + x*y
                misclassified += 1
                mistakes += 1
                if verbose:
                    print('{}-{}: misclassified? y  w: {} '.format(epoch,i, w))
                
            else:
                if verbose:
                    print('{}-{}: misclassified? n  w: {}'.format(epoch,i, w))

        # outside of the examples, inside of epochs loop
        if misclassified == 0:
            final_iter = epoch
            print("\nPerceptron converged after: {} iterations".format(final_iter))
            break
    
    # outside of epochs loop
    if misclassified != 0:
        print("\nPerceptron DID NOT converge until: {} iterations".format(final_iter))
                
    return w, final_iter, mistakes

def perceptron_test(w, X,minus_one=True):
    """
    label y should be 0 or 1.
    """
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X = np.append(ones, X, axis=1)
    
    y_pred = np.sign(X.dot(w))
    
    if minus_one:
        y_pred[y_pred==-1] = 0
    
    return y_pred

def aperceptron_train(data, labels, epochs,verbose=0):
    """data: without bias column
    labels: 1d array
    """
    # data and labels
    X = data
    Y = labels
    
    # change labels 0 to -1 to make h<=0 classification
    Y[Y==0] = -1
    
    # initialize weights
    w = u = np.zeros(X.shape[1] )
    b = beta = 0
    
    # counters    
    final_iter = epochs
    c = 1
    mistakes = 0
    
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
                mistakes += 1
                
        # update counter regardless of good or bad classification        
        c = c + 1
        
        # break loop if w converges
        if misclassified == 0:
            final_iter = epoch
            print("Averaged Perceptron converged after: {} iterations".format(final_iter))
            break
        
        
    if misclassified != 0:
        print("\nAveraged Perceptron DID NOT converge until: {} iterations".format(final_iter))
        
    # prints
    # print("final_iter = {}".format(final_iter))
    # print("b, beta, c , (b-beta/c)= {} {} {} {}".format(b, beta, c, (b-beta/c)))
    # print("w, u, (w-u/c) {} {} {}".format(w, u, (w-u/c)) )

                
    # return w and final_iter
    w = w - u/c
    b = np.array([b- beta/c])
    w = np.append(b, w)
    
    return w, final_iter, mistakes

##============== Kernel (Dual) Perceptron============================
def quadratic_kernel(x, y):
    """
    :Input: two examples x and y.
    
    :Output: the quadratic kernel value computed as (1+xTy)^2. 
    
    """
    Kxy = (1 + np.dot(x,y))**2
    return Kxy

def kperceptron_train(data,labels,epochs,kernel,verbose=False):
    """ 
    Returns: alpha,sv,sv_y,final_iter
    
    """
    X = data
    y = labels
    y = np.array(y)
    y[y==0] = -1
    mistakes = 0
    final_iter = epochs
    
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples, dtype=np.float64)

    # Gram matrix
    # NOTE: kernel Kxy = (1 + np.dot(x,y))**2
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(X[i], X[j])

    # kernel perceptron algorithm
    for epoch in range(epochs):
        misclassified = 0
        
        if verbose:
            print("epoch = {}".format(epoch))
            
            
        for i in range(n_samples):
            h = np.sign(np.sum(alpha * y * K[:,i] ))
            if h != y[i]:
                alpha[i] += 1.0
                misclassified += 1
                mistakes += 1
        
        # inside epochs, outside examples
        if misclassified == 0:
            final_iter = epoch
            print("\nKernel Perceptron converged after: {} iterations".format(final_iter))
            break
    
    # outside of epochs loop
    if misclassified != 0:
        print("\nKernel Perceptron DID NOT converge until: {} iterations".format(final_iter))
    
    
    idx = alpha > 1e-5
    alpha = alpha[idx]
    sv = X[idx]
    sv_y = y[idx]
    
    return alpha,sv,sv_y,final_iter,mistakes

def kperceptron_project(X,kernel,alpha,sv,sv_y):
    
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, svy, sv_ in zip(alpha, sv_y, sv):
            s += a * svy * 1
            s += a * svy * kernel(X[i], sv_)
        y_predict[i] = s
    return y_predict

def kperceptron_test(X,kernel,alpha,sv, sv_y,minus_one=True):
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape
    
    y_pred = np.sign(kperceptron_project(X,kernel,alpha,sv, sv_y))
    
    if minus_one:
        y_pred[y_pred==-1] = 0

    return y_pred

##========================Read example=================================
def read_examples(file_name):
    ldata = np.loadtxt(file_name)
    labels = ldata[:,0]
    data  = ldata[:,1:]
    
    return data, labels


##=========================Read data===================================
def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:,:-1]
    Y = data[:,-1]
       
    return X, Y

def confusion_matrix(y,y_pred):
    # get classes
    classes = list(set(y)) # [0, 1]
    n = len(classes) 
    
    # permutation of [0,1] [0,1] then count the tuples
    cm = [ list(zip(y,y_pred)).count(x) 
                    for x in itertools.product(classes,repeat=2)]
    # 00 01 10 11
    # TN, FN, FP, TP = cm
    
    # Diagonals are True
    #    0    1
    # 0  TN   FN
    # 1  FP   TN
    
    cm = np.array(cm).reshape(n,n)
    
    return cm
