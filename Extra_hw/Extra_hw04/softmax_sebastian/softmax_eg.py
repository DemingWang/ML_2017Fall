#!python
# -*- coding: utf-8 -*-#
"""
Softmax Regression example

@author: Bhishan Poudel

@date: Oct 14, 2017

@email: bhishanpdl@gmail.com

Ref: https://sebastianraschka.com/faq/docs/softmax_regression.html
"""
# Imports
import numpy as np
from numpy import exp, max

def make_onehot(Y, n_sample):
    """Compute class indices to one-hot vectors
    Args:

    Y : column vector of numeric class labels, e.g. Y.T = [0,1,2,2]
    n_sample : no. of samples
    """
    y_onehot = np.zeros((Y.shape[0], n_sample))
    y_onehot[np.arange(Y.shape[0]), Y.T] = 1
    
    
    #   print("y_onehot = {}".format(y_onehot))
    # y_range = np.arange(Y.shape[0])
    # print('y_range = ', y_range)
    # print("Y.T = {}".format(Y.T))
        
    return y_onehot


def softmax(z):
    hyp = np.exp(z-np.amax(z, axis=1, keepdims=True)) # axis 1 is row-wise sum
    s  = hyp / hyp.sum(axis=1, keepdims=True)          # make row sum 1
    
    # zrow = z[0]
    # exp_row = np.exp(zrow)
    # 
    # zrow_max = np.max(z, axis=1)[0]
    # zrow_minus_max = zrow - zrow_max
    # exp_row_minus_rowmax = np.exp(zrow_minus_max)
    # 
    # sum_row_exp = np.sum(ex[0,:])
    
    # print("zrow = {}".format(zrow))
    # print("exp_row = {}".format(exp_row))
    # 
    # print("\nzrow_max = {}".format(zrow_max))
    # print("zrow - max(zrow) = {}".format(zrow - max(zrow)))
    # print("exp_row_minus_rowmax = {}".format(exp_row_minus_rowmax))
    # print("ex = {}".format(ex))
    # 
    # print("\nsum_col_exp = {}".format(sum_row_exp))
    # print("exp_row_minus_rowmax/sum_col_exp = {}".format(exp_row_minus_rowmax/sum_row_exp))
    # print("s = {}".format(s))
    
    return s

def softmax_eg():
    """Run main function."""
    X = np.array([
        [ 0.1, 0.5],
        [1.1,  2.3],
        [-1.1, -2.3],
        [-1.5, -2.5]
    ])

    # print("X = {}".format(X))
    
    # add bias to X
    one = np.ones(X.shape[0])[None].T
    X = np.append(one, X, axis=1)
    # print('x = \n', X) 
    
    # Choose intial weights
    # for 3 classes w has 3 columns.
    # each sample has two features, so w has 2 rows. (x1 = [x10, x11])
    w = np.array([
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3]
    ])

    # Choose bias weights
    # w0 is row vector with 3 elements for 3 classes.
    w0 = np.array([0.01, 0.1, 0.1])[None]
    w = np.append(w0, w, axis=0)
    # print("w0 = {}".format(w0.shape))
    # print('w = \n', w)
    
    # z is the matrix multiplication of X and W  w is NET INPUT
    # z has n_samples row where each row is each sample (columns are classes)
    # z.shape = 4,3  4 samples, 3 classes
    # print("X.shape = {}".format(X.shape)) # 4,3
    # print("w.shape = {}".format(w.shape)) # 3, 3
    
    z = X @ w
    # print("z = \n{}".format(z))
    
    # compute softmax of each values
    s = softmax(z)
    row_max = np.max(s, axis=1, keepdims=True)
    labels = np.argmax(s, axis=1)
    
    print("row_max = {}".format(row_max))
    print("labels = {}".format(labels))
    
def make_onehot_eg():
    Y = np.array([0, 1, 2, 2])[None].T
    print("Y = {}".format(Y))
    
    ohe = make_onehot(Y, 3)
    print("ohe = {}".format(ohe))

def main():
    """Run main function."""
    make_onehot_eg()
    

if __name__ == "__main__":
    main()
