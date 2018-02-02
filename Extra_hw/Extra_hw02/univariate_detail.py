#!python
# -*- coding: utf-8 -*-#
"""
:Topic:
@author: Bhishan Poudel

@date:

@email: bhishanpdl@gmail.com

"""
# Imports


import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt

import scaling
np.set_printoptions(formatter={'float': lambda x: "{:,.4f} ".format(x)})



# Read data matrix X and labels t from text file.
def read_data(file_name):
    # unpack columns to arrays
    *X,t = np.genfromtxt(file_name,unpack=True,dtype=np.float64)

    # give shape to arrays (e.g. X = 50,1 t = 50,1 )
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)

    # debug
    # print("X.shape = {}".format(X.shape))
    # print("t.shape = {}".format(t.shape))

    return X, t


# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X1, t, eta, epochs):
    """Calculate the feature vector w.

    Args:
      X1(matrix): Design matrix with bias column.

      t(column vector): Target vector of shape N * 1.

      eta(float): Learning Rate

      epochs(int): Maximum number of iterations to perform.


    """

    w = np.zeros(2)
    X = X1[:, 1:] # remove bias vector from X1 matrix

    h = None

    for i in range(epochs):
        h = X1 @ w.T
        h = h.reshape(h.shape[0],1)

        # for univariate update w0 and w1 separately
        # w[0] -= eta / len(t) * np.sum(h - t)
        # w[1] -= eta / len(t) * np.sum( (h-t) * X )

        # vectorized method
        # shape h-t = 50, 1
        # shape X1 = 50, 2
        # w shape needed = 1,2
        # w = w - eta / len(t) *  ((h-t).T @ X1)  # shape 1,2 but, [0] shape  2,
        w = w - eta / len(t) *  np.einsum('ij,ik->jk', h-t,X1)

        # debug
        # print("type(w) = {}".format(type(w))) # numpy.ndarray
        # print("w.shape = {}".format(w.shape)) # 1,2 or if [0] used 2,


    # debug
    # print("h.shape = {} t.shape = {} X.shape = {} X[-1] = {}".format(
    #         h.shape, t.shape, X.shape, X[0]))

    # print w
    # print("w = {}".format(w))
    print("w[0] = {}".format(w[0]))
    print("w[0][0] = {:,}".format(w[0][0]))

    # print X
    # print("X = {}".format(X))
    # print("X.shape = {}".format(X.shape))
    # print("h = {}".format(h))

    return w


# Compute RMSE on dataset (X, t).
def compute_rmse(X1, t, w):
    """Compute RMSE.

    Args:
      X1(matrix): Design matrix with bias vector. Shape is N, M+1 e.g. 50, 2

      t(column vector): Target vector. Shape is N, 1 e.g. 50, 1

      w(row vector): Feature vector. Has dimension 1, M+1 e.g. 1,2

    .. note::

        h = X1 @ w.T
        # h = np.einsum('ij,kj->ki', w, X1) # 1,2 50,2 --> 50,1

    """
    w = w.reshape(1, X1.shape[1])
    h = X1 @ w.T
    sse = (h - t) ** 2 # h and t should be both column vector.
    mse = np.mean(sse)
    rmse = np.sqrt(mse)

    # debug
    # print("w.shape = {}".format(w.shape))
    # print("h.shape = {}".format(h.shape))
    # print("t.shape = {}".format(t.shape))
    # print("X1.shape = {}".format(X1.shape))
    # print("h = \n", h)

    return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X1, t, w):
    """Compute the cost function.

    .. math:: J = \\frac{1}{2n} \sum_{i=1}^{n}  \\frac{(h - t)^2}{n}

    """

    # Compute cost
    # N = float(len(t))
    # h = np.dot(X1, w.T)   # h = X1 @ w.T
    # J = np.sum((h - t) ** 2) /2 / N

    # One liner
    J = np.sum((X1 @ w.T - t) ** 2) /2 / float(len(t))


    return J


# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X1, t, w):
    """Compute the gradient of cost function w.r.t. feature vector w.

    Args:
      X1(matrix): Design matrix of shape N, M+1, e.g. 50, 2

      t(column vector): Target column vector of shape N, 1 e.g. 50, 1

      w(row vector): Feature row vector of shape 1, M+1 e.g. 1,2

    """
    w = w.reshape(1, X1.shape[1])
    # h = np.einsum('ij,kj->ki', w, X1) # 1,2 50,2 --> 50,1
    h = X1 @ w.T
    grad = 1 / len(t) *  ( (h-t).T @ X1 )

    return grad


def plot_train_test(X1train,ttrain, X1test,ttest,w):
    plt.style.use('ggplot')
    plt.plot(X1train[:, 1], ttrain,'bo',label='Univariate Train')
    plt.plot(X1test[:, 1], ttest,'g^', label='Univariate Test')
    plt.plot(X1train[:,1], X1train@w.T,'r-',label='Best Fit')
    plt.xlabel('Floor Size (Square Feet)')
    plt.ylabel('House Price (Dollar)')
    plt.title('Univariate Regression')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_epochs_cost(X1, t, eta):
    plt.style.use('ggplot')

    # given epochs
    epochs = np.arange(0, 500+10, step=10)
    costs = [compute_cost(X1, t, (train(X1, t, eta, epoch))) for epoch in epochs]
    print("np.min(costs) = {:.5e}".format(np.min(costs)))

    # get minimum value
    # min_val, min_idx = min((val, idx) for (idx, val) in enumerate(costs))

    min_idx = np.argmin(costs)
    min_val = costs[min_idx]
    print("min_idx, min_val, epochs[min_idx] = {}, {:.5e} ,{}".format(min_idx, min_val,epochs[min_idx]))



    plt.plot(epochs, costs,'bo',label='cost history')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.title('Cost history')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_train_outputs(mean_train,std_train,w,rmse,cost):
    print("               Train Data")
    print("epochs  eta     mean       std       w0          w1          rmse        cost")

    # w0 w1 method
    # print("{}     {}    {}     {:,.2f}    {:,.2f}  {:,.2f}   {:,.2f}   {:,.2f}".format(
    #     500, 0.1, mean_train[0][0], std_train[0][0], w[0], w[1], rmse, cost))

    # vectorized method
    print("{}     {}    {}     {:,.2f}    {:,.2f}  {:,.2f}   {:,.2f}   {:,.2f}".format(
        500, 0.1, mean_train[0][0], std_train[0][0], w[0][0], w[0][1], rmse, cost))


def main():
    """Run main function."""
    parser = argparse.ArgumentParser('Univariate Exercise.')
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../data/univariate',
                        help='Directory for the univariate houses dataset.')
    FLAGS, unparsed = parser.parse_known_args()

    # Read the training and test data.
    Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
    Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

    # Feature Scaling
    mean_train, std_train = scaling.mean_std(Xtrain)
    Xtrain = scaling.standardize(Xtrain,mean_train,std_train)
    Xtest = scaling.standardize(Xtest,mean_train,std_train) # use train mean

    # Add Bias vector to design matrix X.
    X1train = np.append(np.ones_like(ttrain), Xtrain, axis=1)
    X1test = np.append(np.ones_like(ttest), Xtest, axis=1)

    # Get weights or feature vector from training data
    w = train(X1train, ttrain, 0.1, epochs=500)

    # Get RMSE
    rmse = compute_rmse(X1train, ttrain, w)

    # Get Cost
    cost = compute_cost(X1train, ttrain, w)

    # Get gradient
    grad = compute_gradient(X1train, ttrain, w)

    # Print oputputs for train data
    print_train_outputs(mean_train,std_train,w,rmse,cost)


    # Answer to qn1 part d
    plot_train_test(X1train,ttrain, X1test,ttest,w)

    # Answer to qn 1 part b
    X1, t = X1train, ttrain
    eta = 0.1
    plot_epochs_cost(X1, t, eta)


if __name__ == "__main__":
    main()
