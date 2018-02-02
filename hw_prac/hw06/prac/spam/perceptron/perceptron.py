#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 16, 2017

"""
# Imports
import collections
import numpy as np
import itertools

##===================Vanilla Perceptron================================
def perceptron_train(X, Y, epochs,verbose=False):    
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

def perceptron_test(w, X):
    """
    label y should be 0
    -1 or 1.
    """
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X = np.append(ones, X, axis=1)
    
    y_pred = np.sign(X.dot(w))
    
    return y_pred

def read_examples(fdense):
    ldata = np.loadtxt(fdense)
    labels = ldata[:,0]
    data  = ldata[:,1:]
    
    return data, labels

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

def main():
    """Run main function."""
    fdense = '../data/dense_m1.txt'
    X,Y = read_examples(fdense)
    
    epochs = 200
    w, final_iter, mistakes = perceptron_train(X, Y, epochs,verbose=False)

if __name__ == "__main__":
    main()
