#!python
# -*- coding: utf-8 -*-#
"""
:Topic: calculate this.
@author: Bhishan Poudel

@date: Sep 22, 2017

@email: bhishanpdl@gmail.com

Ref: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-2/
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


def main():
    """Run main function."""
    pass

if __name__ == "__main__":
    main()
