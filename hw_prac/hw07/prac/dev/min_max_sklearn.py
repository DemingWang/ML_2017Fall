#!python
# -*- coding: utf-8 -*-#
"""
min-max Scaling

@author: Bhishan Poudel

@date: Nov 17, 2017

"""
# Imports
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, normalize
np.set_printoptions(3)
from numpy import linalg as LA

def min_max_scaler():
    """ 
    X_norm = X - X_min / (X_max - X_min)
    
    Min is calculated for a column by default.
    """
    X = np.array([[1, 19,10], 
                  [0,-10,40],
                  [20,2,4]],
                  dtype='f')  
    
    ## minmax scalar
    scaler = MinMaxScaler()
    # X_norm = scaler.fit_transform(X)     # columns
    X_norm = scaler.fit_transform(X.T).T   # each rows

    print(X_norm)

def min_max_np():
    X = np.array([[1, 19,10], 
                  [0,-10,40],
                  [20,2,4]],
                  dtype='f')  
    
    # numpy method
    # axis 1 is for row wise min-max
    # does not work if keepdims not used.
    X_norm = (X - X.min(axis=1,keepdims=True)) / (X.max(axis=1,keepdims=True) - X.min(axis=1,keepdims=True))
    
    print(X_norm)

def divide_by_max():
    X = np.array([[5, 10,10], 
                  [0,-10,40],
                  [20,2,4]],
                  dtype='f')  
    
    X_norm = preprocessing.normalize(X, axis=0, norm='max')
    
    print(X_norm)

def l1_sum_is_one():
    X = np.array([[5, 10,10], 
                  [0,-10,40],
                  [20,2,4]],
                  dtype='f')  
    
    X_norm = normalize(X, axis=0, norm='l1')
    
    print(X_norm)

def l2_squared_sum_is_one():
    # Two samples, with 3 dimensions.
    # The 2 rows indicate 2 samples, 
    # and the 3 columns indicate 3 features for each sample.
    X = np.asarray([[-1,0,1],
                    [0,1,2]], 
                    dtype=np.float) # Float is needed.
       
    # l2-normalize the samples (rows). 
    X_norm = preprocessing.normalize(X, norm='l2')
     
    # After normalization.
    print (X_norm)
    
    # Square all the elements/features.
    X_squared = X_norm ** 2
    print (X_squared)
    
    # Sum over the rows.
    X_sum_squared = np.sum(X_squared, axis=1)
    print (X_sum_squared)
    
def linalg_norm():
    X = np.array([[1, -2, 3, 6],
              [4, 5, 6, 5],
              [1, 2, 5, 5],
              [4, 5,10,25],
              [5, 2,10,25]])
    
    # l1 = np.abs(X).sum(axis=1)
    # l2 = np.sqrt((X * X).sum(axis=1))
    
    
    l1 = LA.norm(X, axis=1, ord=1)
    l2 = LA.norm(X, axis=1, ord=2)
    
    print(l1)
    print(l2)
    
    # X_norm_l1 =  X / l1.reshape(X.shape[0],1)
    # X_norm_l2 =  X / l2.reshape(X.shape[0],1)
    # print(X_norm_l1)
    # print(X_norm_l2)
       
def main():
    """Run main function."""
    # min_max_scaler()
    # min_max_np()
    
    # normalize row or column to sum 1
    # divide_by_max()
    # l1_sum_is_one()
    
    # np linalg norm
    # linalg_norm()
    
    # l2 norm
    l2_squared_sum_is_one()


if __name__ == "__main__":
    main()
