#!python
# -*- coding: utf-8 -*-#
"""
Preprocess the optical digit data.

@author: Bhishan Poudel

@date: Nov 16, 2017

"""
# Imports
import numpy as np
from sklearn import preprocessing

def split_devel_train(fdata,fdevel,ftrain):
    with open(fdata) as fi, \
        open(fdevel,'w') as fdv, \
        open(ftrain,'w') as ftr:
         
        for i, line in enumerate(fi):
            if i <1000:
                fdv.write(line)
             
            if i >= 1000:
                ftr.write(line)

def split_to10(fdata,nclass):
    """
    fdata is comma separated.
    X_norm = X - X_min / (X_max - X_min)
    
    """
    *X,t = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)  
    
    ## minmax scalar
    scaler = preprocessing.MinMaxScaler()
    X_norm = scaler.fit_transform(X.T).T   # each rows
        
    for num in range(nclass):
        t2 = t.copy()
        t2[t2!=num] = -1
        
        t3 = t2.copy()
        t3[t3==num] = 1
        
        data = np.c_[X_norm,t3]
        np.savetxt(fdata[0:-4]+str(num)+'.txt',data,fmt='%.7g',delimiter=',')

def main():
    """Run main function."""
    
    # data paths
    idir = '../data/optdigits/'
    odir_tr = '../data/optdigits/train/'
    odir_dv = '../data/optdigits/devel/'
    odir_ts = '../data/optdigits/test/'
    
    # data files
    fdata  = idir    + 'optdigits.tra'
    ftrain = odir_tr + 'train.txt'
    fdevel = odir_dv + 'devel.txt'
    ftest  = odir_ts + 'test.txt'
    
    # split train-devel
    split_devel_train(fdata,fdevel,ftrain)
    
    # split into 10 groups
    split_to10(ftrain,10)
    split_to10(fdevel,10)
    split_to10(ftest,10)

if __name__ == "__main__":
    main()
