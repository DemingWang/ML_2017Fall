#!python
# -*- coding: utf-8 -*-#
"""
Feature Scaling

@author: Bhishan Poudel

@date: Nov 17, 2017

"""
# Imports
import numpy as np
from sklearn import preprocessing
np.set_printoptions(3)

def min_max_normalize(fdata,fdata_norm):
    """ 
    X_norm = X - X_min / (X_max - X_min)
    
    Min is calculated for a column by default.
    """
    *X,t = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)  
    
    ## minmax scalar
    scaler = preprocessing.MinMaxScaler()
    X_norm = scaler.fit_transform(X.T).T   # each rows
    
    data = np.c_[X_norm,t]
    np.savetxt(fdata_norm,data)

    return X_norm, t

def main():
    """Run main function."""
    
    # fdata = 'data/data.txt'
    # fdata_norm = 'data/data_norm.txt'
    
    fdata = 'data/train.txt'
    fdata_norm = 'data/train_norm.txt'
    X, t = min_max_normalize(fdata,fdata_norm)
    print("t = ", t)

if __name__ == "__main__":
    main()
