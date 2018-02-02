#!python
# -*- coding: utf-8 -*-#
"""
Univariate Linear Regression using Batch Gradient Descent

@author: Bhishan Poudel

@date:   Oct 4, 2017

@email: bhishanpdl@gmail.com

:Outputs:

  - ../images/multi_cost_history.png

"""
# Imports
import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm,lstsq,inv

import scaling # local import
np.set_printoptions(formatter={'float': lambda x: "{:,.4f} ".format(x)})
np.random.seed(100)


def read_data(file_name):
    *X,t = np.genfromtxt(file_name,unpack=True,dtype=np.float64)
    X,t = np.array(X).T, np.array(t) # change list to array
    t = t.reshape(len(t),1) # make t column vector

    return X, t

def train_norm_eqn(X, t):
    """Train the data using normal equations.

    Args:

      X (array): Design matrix of size (m+1, n). I.e. There are
        m features and one bias column in the matrix X.

      t (column): target column vector

    """
    w = inv(X.T @ X)  @ X.T @ t
    w = np.array(w).reshape(1, len(w)) # make 1d row array

    return w


# Implement gradient descent algorithm to compute w = [w0, w1, w2, w3].
def train_BGD(X1, t, learning_rate, iters):
    """Calculate the feature vector w.

    Args:
    
      X1(matrix): Design matrix with bias column (shape N, M+1 e.g 50,4)

      t(column vector): Target vector (shape N, 1 e.g. 50, 1)

      learning_rate(float): Learning Rate (e.g. 0.1)

      iters(int): Maximum number of iterations to perform.

    Return:
      w (row vector): Parameter row vector of shape (1, features+1)

    """
    w = np.zeros(X1.shape[1]) # Initiliaze w as row vector of zeros.
    h = np.zeros(t.shape) # h has the shape of t, and we also initiliaze it.

    # In batch grad desc, whole design matrix is
    # multiplied with updated feature vector w and we call it h.
    # X1 and t are same but h and w are updated in each iteration.
    for i in range(iters):
        h = X1 @ w.T
        h = h.reshape(h.shape[0],1)

        # update w using batch gradient descent method
        grad = (h-t).T @ X1  # grad_ols = grad of J
        w = w - learning_rate / len(t) *  grad

    # make w row vector
    w = w.reshape(1, X1.shape[1]) # shape = 1, feature + 1
    return w

def train_SGD(X, t, learning_rate, iters) :
    """Calculate weight vector Stochastic Grad Desc.

    .. note::

       Note that X should be normalized before running batch grad descent.

    Args:
      X(matrix): Normalized Design matrix with bias term.

      t(column vector): Normalized Target column vector (shape = 1, samples)

      iters(int): Number of iterations.

      learning_rate(float): Learning rate for gradient descent algorithm.
      
    Return:
      w (row vector): Parameter row vector of shape (1, features+1)

    """
    X=np.array(X)
    t = np.array(t)
    t =t.reshape(len(t),1)
    N = len(t)
    w = np.zeros(X.shape[1])
    w = w.reshape(1,len(w))

    # debug
    # print("\n\n")
    # print("Inside ridge_BGD")
    # print("X.shape = {}".format(X.shape))
    # print("t.shape = {}".format(t.shape))
    # print("w.shape = {}".format(w.shape))
    # print("shrinkage = {}".format(shrinkage))
    # print("iters = {}".format(iters))
    # print("learning_rate = {}".format(learning_rate))
    # Initiliaze variables
    Xj, tj, hj = [], 0, 0
    iters = int(iters)
    for i in range(0, iters):
        # shuffle the data
        np.random.seed(100)
        perm_idx = np.random.permutation(X.shape[0])
        X = X[perm_idx]
        t = t[perm_idx]

        # update w line by line
        for j in range(X.shape[0]):

            Xj = X[j,:]
            hj = Xj @ w.T
            tj = t[j]

            # reshape
            Xj = Xj.reshape(X.shape[1], 1)
            tj = tj.reshape(1,1)
            hj = hj.reshape(1,1)

            grad_ols =  (hj-tj) @ Xj.T / N
            w = w - learning_rate * grad_ols

    # debug
    # print("Xj.shape = {}".format(Xj.shape)) # 2,1  x has one bias term.
    # print("w.shape = {}".format(w.shape))   # 1,2  w = [w0, w1]
    # print("hj.shape = {}".format(hj.shape)) # 1,1  h and t are single values
    # print("tj.shape = {}".format(tj.shape)) # 1,1
    # print("Xj = {}".format(Xj))
    # print("tj = {}".format(tj))

    # make w row vector
    w = w.reshape(1, X.shape[1]) # shape = 1, feature + 1
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

    .. math:: J = \\frac{1}{2N} \sum_{n=1}^{N}  (h_n - t_n)^2

    """

    # Compute cost
    N = float(len(t))
    h = X1 @ w.T
    J = np.sum((h - t) ** 2) /2 / N

    return J


# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X1, t, w):
    """Compute the gradient of cost function w.r.t. feature vector w.

    Args:
      X1(matrix): Design matrix of shape N, M+1, e.g. 50, 2

      t(column vector): Target column vector of shape N, 1 e.g. 50, 1

      w(row vector): Feature row vector of shape 1, M+1 e.g. 1,2

    Return:
      grad (row vector): Gradient row vector of shape 1, M+1 same as shape of w.
      
      .. math:: \\nabla_w J = [ dJ/w_0 \\quad dJ/w_1 \\quad dJ/w_2 \\quad dJ/w_M]
      
      gradient::
      
        grad = 1 / len(t) *  ( (h-t).T @ X1 )

    """
    w = w.reshape(1, X1.shape[1])
    h = X1 @ w.T
    grad = 1 / len(t) *  ( (h-t).T @ X1 )

    return grad


def print_train_outputs(epochs, learning_rate, mean_train,std_train,w,rmse,cost):
    print("               Train Data")
    print("epochs  learning_rate     mean        std       w0          w1          w2          w3        rmse        cost")
    print("{}     {}    {}     {:,.2f}    {:,.2f}  {:,.2f}   {:,.2f}  {:,.2f}   {:,.2f}   {:,.2f}".format(
        epochs,    learning_rate,   mean_train[0][0], std_train[0][0], w[0][0], w[0][1],w[0][2],w[0][3], rmse, cost))

def plot_cost_history(X1, t, iters,step, learning_rate):
    plt.style.use('ggplot')

    # given epochs
    iters_lst = np.arange(0, iters+step, step)
    costs_bgd = [compute_cost(X1, t, (train_BGD(X1, t, learning_rate, itr))) for itr in iters_lst]
    costs_sgd = [compute_cost(X1, t, (train_SGD(X1, t, learning_rate, itr))) for itr in iters_lst]

    plt.plot(iters_lst, costs_bgd,'bo',label='cost history GD')
    plt.plot(iters_lst, costs_sgd,'ro',label='cost history SGD',ms=2)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.title('Multivariate Cost history')
    plt.legend()
    plt.grid(True)
    plt.ylim(min(costs_bgd)*0.99,costs_bgd[4])
    plt.tight_layout()
    plt.savefig('../images/multi_cost_history.png')
    plt.show()

def main():
    """Run main function."""
    parser = argparse.ArgumentParser('Multivariate Exercise.')
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../data/multivariate',
                        help='Directory for the multivariate houses dataset.')
    FLAGS, unparsed = parser.parse_known_args()

    # Read the training and test data.
    Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
    Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

    # Normalize and add bias term to Train data
    mean_train, std_train = scaling.mean_std(Xtrain)
    Xtrain = scaling.standardize(Xtrain,mean_train,std_train)
    X1train = np.append(np.ones_like(ttrain), Xtrain, axis=1)


    # Hyperparameters
    iters, step, learning_rate = 500, 10, 0.1


    # Compare w from normal eqn and BGD
    w_norm_eqn = train_norm_eqn(X1train, ttrain)
    w_bgd      = train_BGD(X1train, ttrain, learning_rate, iters)
    print("w_norm_eqn = {}".format(w_norm_eqn))
    print("w_bgd      = {}".format(w_bgd))


    # Normalize and add bias term to Test data
    Xtest = scaling.standardize(Xtest,mean_train,std_train) # XXX
    X1test = np.append(np.ones_like(ttest), Xtest, axis=1)


    # Parameters for Train
    #
    #
    w = train_BGD(X1train, ttrain, learning_rate, iters)
    rmse = compute_rmse(X1train, ttrain, w)
    cost = compute_cost(X1train, ttrain, w)
    grad = compute_gradient(X1train, ttrain, w)
    print_train_outputs(iters, learning_rate, mean_train,std_train,w,rmse,cost)

    # plots
    plot_cost_history(X1train, ttrain, iters,step, learning_rate)

    # Compare w from BGD and SGD
    #
    w_norm_eqn = train_norm_eqn(X1train, ttrain)
    iters = 500
    abs_diff_min_lst = []
    for i in np.arange(iters):
        w_sgd = train_SGD(X1train, ttrain, learning_rate, i)
        print("\n")
        print("i = {}".format(i))
        print("w_norm_eqn        = {}".format(w_norm_eqn))
        print('iters = {} w_BGD = {}'.format(iters,w))
        print("iters = {} w_SGD = {}".format(i, w_sgd))

        abs_diff_min = [ abs(w_sgd[0][0]-w_norm_eqn[0][0]) +
             abs(w_sgd[0][1]-w_norm_eqn[0][1]) +
             abs(w_sgd[0][2]-w_norm_eqn[0][2]) +
             abs(w_sgd[0][3]-w_norm_eqn[0][3]) ][0]

        abs_diff_min_lst.append(abs_diff_min)
        print("abs_diff_min = {}".format(abs_diff_min))
        print("len(abs_diff_min_lst) = {}".format(len(abs_diff_min_lst)))
        print('np.argmin(abs_diff_min_lst) = ', np.argmin(abs_diff_min_lst))


if __name__ == "__main__":
    main()
