#!python
# -*- coding: utf-8 -*-#
"""
Multiclass perceptron

@author: Bhishan Poudel

@date: Nov 18, 2017

"""
# Imports
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import perceptron


def read_data(fdata):
    *X,t = np.genfromtxt(fdata,unpack=True,dtype='f')
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    
    return X,t

def multiclass_perceptron(data_pth):
    # create big hypothesis matrix
    fdata = data_pth + '1.txt'
    X, t = read_data(fdata)
    nclass = 10
    hyp = np.zeros([X.shape[0], nclass ])
    print("X.shape = {}".format(X.shape))
    print("t.shape = {}".format(t.shape))
    print("hyp.shape = {}".format(hyp.shape))
    print("\n")
    
    # instanciate a model
    net = perceptron.Perceptron(max_iter=1, verbose=0, 
                                random_state=100, fit_intercept=True)
    
    # fit the model
    for i in range(10):
        fdata = data_pth+ str(i) + '.txt'
        X, t  = read_data(fdata)
        net.fit(X,t)
        w = net.coef_
        h = X.dot(w.T).ravel()    
        hyp[:,i] = h
        
        if i==0:
            print("hyp[:,0] = {}".format(hyp[:,0]))
            print("hyp0")
            print(hyp[0])
    
    ##get the maximum argvalue
    t_pred = np.argmax(hyp,axis=1)
    correct = np.sum(t_pred==t)
    
    print("hyp.shape = {}".format(hyp.shape))
    print("t_pred.shape = {}".format(t_pred.shape))
    print("correct = {}".format(correct))
    print("t_pred = {}".format(t_pred))
    
    
def main():
    """Run main function."""
    data_pth = 'data/prac/data'
    
    # fdata = '../data/optdigits/train/train0.txt'
    # data_pth = '../data/optdigits/train/train'
    multiclass_perceptron(data_pth)

if __name__ == "__main__":
    main()
