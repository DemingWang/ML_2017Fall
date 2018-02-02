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
    *X,y = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X,y,y = np.array(X).T, np.array(y), y.reshape(len(y),1) 
    
    ## minmax scalar
    scaler = preprocessing.MinMaxScaler()
    X_norm = scaler.fit_transform(X.T).T   # each rows
    
    data = np.c_[X_norm,y]
    np.savetxt(fdata_norm,data,fmt='%.5f')

    return X_norm, y

def main():
    """Run main function."""
    
    fdata = 'data/prac/data.txt'
    fdata_norm = 'data/prac/data_norm.txt'
    X, y = min_max_normalize(fdata,fdata_norm)
    
    fdata = 'data/prac/data.txt'
    fdata_norm = 'data/prac/data_norm.txt'
    X, y = min_max_normalize(fdata,fdata_norm)
    print("y = ", y)

if __name__ == "__main__":
    main()
