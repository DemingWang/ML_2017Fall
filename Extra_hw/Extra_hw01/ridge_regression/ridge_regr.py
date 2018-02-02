#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 21, 2017
# Last update :
###########################################################################
"""
:Topic: Ridge Regression

:Ref: http://hyperanalytic.net/ridge-regression
"""
# Imports
import numpy as np

def ridge_regression(x_train, y_train, lam):

    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones,X))
    y = np.array(y_train)

    Xt = np.transpose(X)
    lambda_identity = lam*np.identity(len(Xt))
    theInverse = np.linalg.inv(np.dot(Xt, X)+lambda_identity) # lamda N I , N is missing!!!
    w = np.dot(np.dot(theInverse, Xt), y)
    return w

def main():
    """Run main function."""
    x_train = np.arange(0,100,step=0.02)
    y_train = x_train * 2 + 1
    w = ridge_regression(x_train, y_train, lam=0.001)
    print(w)

if __name__ == "__main__":
    main()
