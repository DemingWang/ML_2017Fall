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

def split_to10(fdata):
    """ 
    X_norm = X - X_min / (X_max - X_min)
    
    Min is calculated for a column by default.
    """
    *X,t = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)  
    
    ## minmax scalar
    scaler = preprocessing.MinMaxScaler()
    X_norm = scaler.fit_transform(X.T).T   # each rows
        
    for num in range(10):
        t2 = t.copy()
        t2[t2!=num] = -1
        
        t3 = t2.copy()
        t3[t3==num] = 1
        
        data = np.c_[X_norm,t3]
        np.savetxt(fdata[0:-4]+str(num)+'.txt',data,fmt='%.3f')

def main():
    """Run main function."""
    
    fdata = 'data/prac/data.txt'
    # fdata_norm = 'data/data_norm.txt'
    
    # fdata = 'data/train.txt'
    split_to10(fdata)


if __name__ == "__main__":
    main()
