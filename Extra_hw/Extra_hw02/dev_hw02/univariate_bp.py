#!python
# -*- coding: utf-8 -*-#
"""
:Topic: Univariate Linear Regression using Batch Gradient Descent

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
    *X,t = np.genfromtxt(file_name,unpack=True,dtype=np.float64)
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)

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
    w = np.zeros(X1.shape[1])
    X = X1[:, 1:] # remove bias vector from X1 matrix
    h = np.zeros(t.shape)

    for i in range(epochs):
        h = X1 @ w.T
        h = h.reshape(h.shape[0],1)

        # for univariate update w0 and w1 separately
        w[0] -= eta / len(t) * np.sum(h - t)
        w[1] -= eta / len(t) * np.sum( (h-t) * X )

    return w


# Compute RMSE on dataset (X, t).
def compute_rmse(X1, t, w):
    """Compute RMSE.

    Args:

      X1(matrix): Design matrix with bias vector. Shape is N, M+1 e.g. 50, 2

      t(column vector): Target vector. Shape is N, 1 e.g. 50, 1

      w(row vector): Feature vector. Has dimension 1, M+1 e.g. 1,2

    """
    w = w.reshape(1, X1.shape[1])
    h = X1 @ w.T
    sse = (h - t) ** 2 # h and t should be both column vector.
    mse = np.mean(sse)
    rmse = np.sqrt(mse)

    return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X1, t, w):
    """Compute the cost function.

    .. math:: J = \\frac{1}{2n} \sum_{i=1}^{n}  \\frac{(h - t)^2}{n}

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
    
      X1(matrix): Design matrix with bias term (shape = N, M+1, e.g. 50, 2)

      t(column vector): Target column vector (shape = N, 1 e.g. 50, 1)

      w(row vector): Feature row vector (shape = 1, M+1 e.g. 1,2)

    """
    w = w.reshape(1, X1.shape[1])
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
    plt.title('Univariate Regression Batch Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/univ_BGD.png')
    # plt.show()
    plt.close()




def plot_cost_history(X1, t, epochs,step, eta):
    plt.style.use('ggplot')

    # given epochs
    epochs_lst = np.arange(0, epochs+step, step)
    costs = [compute_cost(X1, t, (train(X1, t, eta, epoch))) for epoch in epochs_lst]
    min_idx = np.argmin(costs)
    print("np.min(costs) = {:.5e}".format(np.min(costs)))
    print("epochs_lst[min_idx] = {}".format(epochs_lst[min_idx]))

    # plot
    plt.plot(epochs_lst, costs,'bo',label='cost history')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.title('Univariate Cost history BGD ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/univ_cost_history_BGD.png')
    plt.show()
    plt.close()

def print_train_outputs(epochs, eta, mean_train,std_train,w,rmse,cost):
    """Note: in vectorized method w[1] is replaced by w[0][1] and so on"""
    print("               Train Data")
    print("epochs  eta     mean       std       w0          w1          rmse        cost")
    print("{}     {}    {}     {:,.2f}    {:,.2f}  {:,.2f}   {:,.2f}   {:,.2f}".format(
        epochs, eta, mean_train[0][0], std_train[0][0], w[0], w[1], rmse, cost))


## ====================== Extra ==============================================

def plot_test_only(X1train, ttrain, X1test,ttest, w, epochs,cost):
    """Create png files to fit test data.

    """
    print("epochs = {}".format(epochs))
    plt.style.use('ggplot')
    plt.plot(X1test[:, 1], ttest,'g^', label='Test \nepochs = {:03d} \nJ = {:.2e}'.format(epochs,cost))
    plt.plot(X1train[:,1], X1train@w.T,'r-',label='Best Fit')
    plt.xlabel('Floor Size (Square Feet)')
    plt.ylabel('House Price (Dollar)')
    plt.title('Univariate Regression Batch Gradient Descent')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../Extra/test_images/test_{:03d}.png'.format(epochs))
    plt.close()

def create_gif(X1train, ttrain, X1test,ttest, w, epochs,cost,eta):
    """We create 70 png files and use ImageMagick command to create gif.

    ```convert -loop 0 -delay 100 test/test*.png cost_history.gif```

    In this case Cost function J does not decreases after 70 epochs, so I
    create only 70 png files.

    """
    for epochs in np.arange(70):
        w = train(X1train, ttrain, eta, epochs)
        cost = compute_cost(X1test, ttest, w)
        plot_test_only(X1train, ttrain, X1test,ttest, w, epochs,cost)
## ====================== Extra End ==========================================


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

    # Normalize and add bias term to Train data
    mean_train, std_train = scaling.mean_std(Xtrain)
    Xtrain = scaling.standardize(Xtrain,mean_train,std_train)
    X1train = np.append(np.ones_like(ttrain), Xtrain, axis=1)

    # Normalize and add bias term to Test data
    Xtest = scaling.standardize(Xtest,mean_train,std_train) # XXX
    X1test = np.append(np.ones_like(ttest), Xtest, axis=1)
    X1, t = X1test, ttest

    # Hyperparameters
    epochs, step, eta = 500, 10, 0.1


    # Get w from Train and use it on Test
    w = train(X1train, ttrain, eta, epochs)



    # Parameters for Test
    rmse = compute_rmse(X1, t, w)
    cost = compute_cost(X1, t, w)
    grad = compute_gradient(X1, t, w)
    print_train_outputs(epochs, eta, mean_train,std_train,w,rmse,cost)

    # plots
    plot_train_test(X1train,ttrain, X1test,ttest,w)
    plot_cost_history(X1, t, epochs,step, eta)

    # Extra
    # create gifs
    create_gif(X1train, ttrain, X1test,ttest, w, epochs,cost,eta)


if __name__ == "__main__":
    main()
