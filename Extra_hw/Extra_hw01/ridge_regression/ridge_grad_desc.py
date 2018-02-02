#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 21, 2017
# Last update :
###########################################################################
"""
:Topic: Ridge Regression With Grad descent

:Ref: http://hyperanalytic.net/ridge-regression

:Algorithm:

  βj := βj - α[(1/m)Σ(yi-f(xi))(xi)+(λ/m)βj]

"""
# Imports
import numpy as np
from sklearn import datasets
from scipy import stats

def RidgeGradientDescent(x, y, alpha, iters, L):
    x=np.matrix(x)
    y=np.matrix(y).transpose()
    m, n = np.shape(x)
    beta = np.matrix(np.ones(n)).transpose()
    XT = x.transpose()
    for i in range(0, iters):
        y_hat = np.dot(x, beta)
        residuals = y_hat - y
        MSE = (residuals.transpose()*residuals)/len(x)
        print ("iteration:", i, "MSE:", MSE)
        ols_gradient = np.dot(XT, residuals) / m
        beta = beta - alpha * (ols_gradient + (L/m)*beta)
    return beta

def main():
    """Run main function."""
    diabetes = datasets.load_diabetes()

    X = diabetes.data
    y = diabetes.target
    intercept = np.ones(len(X))
    X = np.append(intercept, X)
    X = np.reshape(X,(442,11))

    Z = stats.zscore(X, axis=0)
    Y = stats.zscore(y)

    w = RidgeGradientDescent(Z,Y,.1,5000,.1)
    print (w.T)

if __name__ == "__main__":
    main()
