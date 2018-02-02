#!python
# -*- coding: utf-8 -*-#
"""
Feature Scaling

@author: Bhishan Poudel

@date: Nov 17, 2017

"""
# Imports
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
np.set_printoptions(3)

def split_train_test(fdata):
    # file names
    ftrain = fdata[0:-9]+'_train_orig.txt'
    ftest = fdata[0:-9]+'_test.txt'
    
    *X,y = np.genfromtxt(fdata,unpack=True,dtype='str',delimiter=',')
    X,y,y = np.array(X).T.astype(float), np.array(y), y.reshape(len(y),1) 

    # encode names
    y[0:50] = 0
    y[50:100] = 1
    y[100:150] = 2
    y = np.array(y).astype(int)
    np.savetxt(fdata[0:-9]+'.txt',np.c_[X,y],fmt='%g')
    
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
    np.savetxt(ftrain,np.c_[X_train,y_train],fmt='%g')
    np.savetxt(ftest,np.c_[X_test,y_test],fmt='%g')
    # print("y_train.shape = {}".format(y_train.shape))
    # print("y_test.shape = {}".format(y_test.shape))
    
def split_train_devel(ftrain_orig):
    # file names
    ftrain = ftrain_orig[0:-15]+'_train.txt'
    fdevel = ftrain_orig[0:-15]+'_devel.txt'
    
    *X,y = np.genfromtxt(ftrain_orig,unpack=True,dtype='f',delimiter=' ')
    X,y,y = np.array(X).T.astype(float), np.array(y), y.reshape(len(y),1)
    
    # split train devel
    X_train, X_devel, y_train, y_devel = train_test_split(X, y, test_size=0.4, random_state=5)
    np.savetxt(ftrain,np.c_[X_train,y_train],fmt='%g')
    np.savetxt(fdevel,np.c_[X_devel,y_devel],fmt='%g')
    # print("y_train.shape = {}".format(y_train.shape))
    # print("y_devel.shape = {}".format(y_devel.shape))

def split_classes(fdata):
    # file names
    ftrain = fdata[0:-15]+'_train.txt'
    fdevel = fdata[0:-15]+'_devel.txt'
    
    *X,y = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=' ')
    X,y,y_ = np.array(X).T.astype(float), np.array(y), y.reshape(len(y),1).astype(int)
    
    # iris0
    num = 0
    y = y_.copy()
    y[y!=num] = -1  # it should be before y==num
    y[y==num] = 1
    
    data = np.c_[X,y]
    np.savetxt(fdata[0:-4]+str(num)+'.txt',data,fmt='%g')
    
    # iris1
    num = 1
    y = y_.copy()
    y[y!=num] = -1
    y[y==num] = 1
    y = np.array(y).astype(int)
    
    data = np.c_[X,y]
    np.savetxt(fdata[0:-4]+str(num)+'.txt',data,fmt='%g')
    
    # iris2
    num = 2
    y = y_.copy()
    y[y!=num] = -1
    y[y==num] = 1
    y = np.array(y).astype(int)
    
    data = np.c_[X,y]
    np.savetxt(fdata[0:-4]+str(num)+'.txt',data,fmt='%g')

def main():
    """Run main function."""
    
    fdata = '../data/iris/orig/iris_orig.txt'
    ftrain_orig = '../data/iris/orig/iris_train_orig.txt'
    ftrain = '../data/iris/train/iris_train.txt'
    fdevel = '../data/iris/devel/iris_devel.txt'
    ftest = '../data/iris/test/iris_test.txt'
    
    split_train_test(fdata)
    split_train_devel(ftrain_orig)
    
    split_classes(ftrain)
    split_classes(fdevel)
    split_classes(ftest)


if __name__ == "__main__":
    main()
