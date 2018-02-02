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

:Algorithm::

  grad_ols =  (h-t).T @ X / N
  grad_ridge = (grad_ols + shrinkage * w )  # shrinkage /N for some cases.
  w = w - l_rate * grad_ridge

"""
# Imports
import numpy as np
from sklearn import datasets
from scipy import stats

def ridge_batch_grad_desc(X, t, shrinkage, iters, l_rate):
    """Calculate weight vector using Ridge Regression L2 norm.

    Args:
      X(matrix): Design matrix with bias term.

      t(column vector): Target column vector (shape = 1, samples)

      shrikage(float): L2 regularization shrikage hyper parameter.

      iters(int): Number of iterations.

      l_rate(float): Learning rate for gradient descent algorithm.

    """
    X=np.array(X)
    t = np.array(t)
    t =t.reshape(len(t),1)
    N = len(t)
    w = np.ones(X.shape[1])
    w = w.reshape(1,len(w))

    print("x.shape = {}".format(X.shape))
    print("t.shape = {}".format(t.shape))
    print("w.shape = {}".format(w.shape))
    print("shrinkage = {}".format(shrinkage))
    print("iters = {}".format(iters))
    print("l_rate = {}".format(l_rate))
    for i in range(0, iters):
        h = X @ w.T
        MSE = np.square(h - t).mean()
        print ("iteration:", i, "MSE:", MSE)
        grad_ols =  (h-t).T @ X / N
        grad_ridge = (grad_ols + shrinkage * w )  # shrinkage /N for some cases.
        w = w - l_rate * grad_ridge

    # make w row vector
    w = w.reshape(1, X.shape[1]) # shape = 1, feature + 1
    return w

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

    w = ridge_batch_grad_desc(Z,Y,.1,5000,.1)
    print (w)

if __name__ == "__main__":
    main()
