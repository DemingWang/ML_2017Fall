#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 17, 2017

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import perceptron
from sklearn.metrics import confusion_matrix

def read_data(fdata):
    data = np.loadtxt(fdata)
    X  = data[:,0:-1]
    Y = data[:,-1]
    
    return X, Y


def perc_sklearn():
    # Data
    ftrain = '../data/prac/train.txt'
    ftest = '../data/prac/test.txt'
    X_train, t_train = read_data(ftrain)
    X_test, t_test = read_data(ftest)
     
    # Create the model
    net = perceptron.Perceptron(max_iter=1, random_state=100, n_jobs=-1,fit_intercept=1 )
    net.fit(X_train,t_train)
    
    w = net.coef_
    print('w = ', w)
    h = w @ X_test.T
    print("h = \n", h)
    h = np.sign(h)
    
    t_pred = net.predict(X_test)
    t_pred2 = h
    
    print("t_pred - t_pred2 = {}".format(t_pred - t_pred2))
     
    # Print to have a look
    # print(confusion_matrix(t_test,t_pred))  

def main():
    """Run main function."""
    perc_sklearn()


if __name__ == "__main__":
    main()
