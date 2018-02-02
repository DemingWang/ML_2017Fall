#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 21, 2017
# Last update :
###########################################################################
"""
:Topic: Regularized Ridge Regression

:Ref: https://statcatinthehat.wordpress.com/2014/07/16/regularized-regression-ridge-in-python-part-2/

"""
# Imports
import numpy as np
from numpy import linalg
from sklearn import datasets, preprocessing
from scipy import stats


# For np.matrix objects
# It has certain special operators, such as * (matrix multiplication) and ** (matrix power).
# * is not elementwise operator, it is matrix multiplication.

def RidgeAnalytical(x,y,L):
    x=np.array(x)
    y = np.array(y)
    y = y.reshape(len(y),1)
    XTX = x.T @ x
    XTY = x.T @ y
    LI = L * np.eye(len(XTX))

    print("\n")
    print("Inside ridge analysis")
    print("x.shape = {}".format(x.shape))
    print("y.shape = {}".format(y.shape))
    print("XTX.shape = {}".format(XTX.shape))
    print("XTY.shape = {}".format(XTY.shape))
    print("LI.shape = {}".format(LI.shape))

    # don't regularize bias term
    LI[0,0]=0 # Don't regularize bias

    beta = np.linalg.inv(XTX-LI) @ XTY # weight vector w or beta
    y_hat = x @ beta
    residuals = y_hat - y
    SSE = residuals.transpose() @ residuals
    MSE = SSE/len(x)
    print ("the parameter estimates are:")
    print (beta)
    print ("the mean squared error is:" , MSE)
    print("beta.shape = {}".format(beta.shape))
    print("y_hat.shape = {}".format(y_hat.shape))


# def RidgeAnalytical(x,y,L):
#     x=np.matrix(x)
#     y=np.matrix(y).transpose()
#     XTX = x.transpose()*x
#     XTY = x.transpose()*y
#     LI = L*np.matrix(np.identity(len(XTX)))
#
#     print("len(XTX) = {}".format(len(XTX)))
#     print("LI.shape = {}".format(LI.shape))
#     LI[0,0]=0 # Don't regularize bias
#     beta = np.linalg.inv(XTX-LI)*XTY # weight vector w or beta
#     y_hat = np.dot(x,beta)  # h = wT X
#     residuals = y_hat - y
#     SSE = residuals.transpose()*residuals
#     MSE = SSE/len(x)
#     print ("the parameter estimates are:")
#     print (beta)
    # print ("the mean squared error is:" , MSE)

def main():
    """Run main function."""

    diabetes = datasets.load_diabetes()
    print("diabetes.data.shape = ", diabetes.data.shape)
    print("diabetes.target.shape = ", diabetes.target.shape)
    X = diabetes.data
    y = diabetes.target
    intercept = np.ones(len(X))
    X = np.append(intercept, X)

    print("X.shape = ", X.shape)
    print("y.shape = ", y.shape)
    print("y.shape[0] = ", y.shape[0])
    print("X.data.shape = ", X.data.shape)
    data_shape = (y.shape[0], 1+diabetes.data.shape[1])
    print("data_shape = ", data_shape)
    X = np.reshape(X,data_shape)

    Z = stats.zscore(X, axis=0)
    Y = stats.zscore(y)

    # run ridge analysis
    RidgeAnalytical(Z,Y,.1)

if __name__ == "__main__":
    main()
