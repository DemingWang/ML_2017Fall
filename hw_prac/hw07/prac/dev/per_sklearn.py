#!python
# -*- coding: utf-8 -*-#
"""
Multi-class Perceptron sklearn

@author: Bhishan Poudel

@date: Nov 14, 2017
http://stamfordresearch.com/scikit-learn-perceptron/
"""
# Imports
import numpy as np
from sklearn.linear_model import perceptron

def read_data(fdata):
    *X,t = np.genfromtxt(fdata,unpack=True,dtype='f')
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    
    return X,t

def perc_sklearn(X_train,t_train,X_test,t_test):
    # Create the model
    net = perceptron.Perceptron(max_iter=100, random_state=100,
                                fit_intercept=True)
    net.fit(X_train,t_train)
   
    # Coeefficients
    w = net.coef_
    print('w.shape = ', w.shape)
    
    # # predictions
    # t_pred = net.predict(X_test)
    # correct = np.sum(t_pred==t_test)
    # accuracy = correct / len(t_pred)
    # 
    # print("accuracy = {}".format(accuracy))

def kperceptron_sklearn(X_train,t_train,X_test,t_test):
    # Create the model
    net = perceptron.Perceptron(max_iter=100, random_state=100,
                                fit_intercept=True)
    net.fit(X_train,t_train)
   
    # Coeefficients
    w = net.coef_
    print('w.shape = ', w.shape)
    
    # # predictions
    # t_pred = net.predict(X_test)
    # correct = np.sum(t_pred==t_test)
    # accuracy = correct / len(t_pred)
    # 
    # print("accuracy = {}".format(accuracy))

def main():
    """Run main function."""
    # X_train, t_train = read_data('../data/prac/train.txt')
    # X_test, t_test = read_data('../data/prac/test.txt')
    # perc_sklearn(X_train,t_train,X_test,t_test)
    
    
    X_train, t_train = read_data('data/prac/data0.txt')
    X_test, t_test = read_data('data/prac/data1.txt')
    
    print("X_train.shape = {}".format(X_train.shape))
    print("t_train.shape = {}".format(t_train.shape))
    perc_sklearn(X_train,t_train,X_test,t_test)
    



if __name__ == "__main__":
    main()
