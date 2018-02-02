#!python
# -*- coding: utf-8 -*-#
"""
Univariate Linear Regression using Batch Gradient Descent

@author: Bhishan Poudel

@date: Oct 3, 2017

@email: bhishanpdl@gmail.com

:Outputs:

  - ../images/univ_BGD.png
  - ../images/univ_cost_history_BGD.png


"""
# Imports
import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv

import scaling

np.set_printoptions(formatter={'float': lambda x: "{:,.4f} ".format(x)})
np.random.seed(100)


# Results for random seed 100
# epochs = 500 w_BGD = [[254,450.0000  93,308.9201 ]]
# epochs = 100 w_SGD = [[254,433.4433  93,299.5155 ]]


# Read data matrix X and labels t from text file.
def read_data(file_name):
    *X,t = np.genfromtxt(file_name,unpack=True,dtype=np.float64)
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)

    return X, t


def train_norm_eqn(X1, t):
    """Train the data and return the weights w.

    Args:

      X1 (array): Design matrix of size (m+1, n). I.e. There are
        m features and one bias column in the matrix X1.

      t (column): target column vector

    Return:
      w(row vector): Weight or Parameter vector (2d row array).
      
    .. math:: w = [w_0, w_1, ..., w_M]

    """
    w = inv(X1.T @ X1) @ X1.T   @t
    w = np.array(w).reshape(1, len(w)) # make 1d row array

    return w


# Implement gradient descent algorithm to compute w = [w0, w1].
def train_BGD(X1, t, learning_rate, iters):
    """Calculate the feature vector w.

    Args:

      X1(matrix): Design matrix with bias column.

      t(column vector): Target vector of shape N * 1.

      learning_rate(float): Learning Rate

      iters(int): Maximum number of iterations to perform.


    """
    w = np.zeros(X1.shape[1])
    X = X1[:, 1:] # remove bias vector from X1 matrix (for w1)
    h = np.zeros(t.shape)



    # print("Univariate Method")
    iters = int(iters)
    for i in range(iters):
        h = X1 @ w.T
        h = h.reshape(h.shape[0],1)

        # for univariate update w0 and w1 separately
        w[0] -= learning_rate / len(t) * np.sum(h - t)
        w[1] -= learning_rate / len(t) * np.sum( (h-t) * X )


    # Compare with multivariate method
    # multivariate method
    multivariate = 0
    if multivariate:
        # print("Univariate Method is overridden.")
        # print("Multivariate Method")
        for i in range(iters):
            h = X1 @ w.T
            h = h.reshape(h.shape[0],1)

            # update w using batch gradient descent method
            w = w - learning_rate / len(t) *  ((h-t).T @ X1)


    # XXX WARNING:
    # if we don't make w as row vector, we will get different cost value
    # from univariate and multivariate method.
    #
    # make w row vector
    # w = np.array(w)
    # w = w.reshape(1, X1.shape[1])

    return np.array(w).reshape(1, X1.shape[1])

def train_SGD(X, t, learning_rate, iters):
    """Calculate weight vector Stochastic Grad Desc.

    .. note::

       Note that X should be normalized before running batch grad descent.

    Args:
      X(matrix): Normalized Design matrix with bias term.

      t(column vector): Normalized Target column vector (shape = 1, samples)

      learning_rate(float): Learning rate for gradient descent algorithm.

      iters(int): Number of iterations.

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
    Xj, tj, hj = 0, 0, 0
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

      w(row vector): Parameter vector. Has dimension 1, M+1 e.g. 1,2
      
    Return:
      rmse (float): RMSE value.

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

    .. math:: J = \\frac{1}{2N} \sum_{n=1}^{N}  (h_n - t_n)^2

    """
    # Compute cost
    N = float(len(t))
    h = X1 @ w.T
    J = np.sum((h - t) ** 2) /2 / N

    # debug
    # w is obtained from train function.
    # print('from compute cost w = ', w)
    # print('from compute cost w.shape = ', w.shape)


    return J


# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X1, t, w):
    """Compute the gradient of cost function w.r.t. feature vector w.

    Args:

      X1(matrix): Design matrix with bias term (shape = N, M+1, e.g. 50, 2)

      t(column vector): Target column vector (shape = N, 1 e.g. 50, 1)

      w(row vector): Feature row vector (shape = 1, M+1 e.g. 1,2)

    Return:
      grad (row vector): Gradient row vector (shape = 1, M+1, same as shape of w)
      
    .. math:: \\nabla_w J = [dJ/w_0 \\quad dJ/w_1 ... \\quad dJ/w_M ]

    """
    w = w.reshape(1, X1.shape[1])
    h = X1 @ w.T
    grad = 1 / len(t) *  ( (h-t).T @ X1 )

    # print("grad.shape = {}".format(grad.shape)) # (1,2) shape same as of w.
    return grad


def plot_train_test(X1train,ttrain, X1test,ttest,w_bgd, w_sgd):
    plt.style.use('ggplot')
    plt.plot(X1train[:, 1], ttrain,'bo',label='Univariate Train')
    plt.plot(X1test[:, 1], ttest,'g^', label='Univariate Test')
    plt.plot(X1train[:,1], X1train @ w_bgd.T,'b-',label='Model fit GD',lw=8)
    plt.plot(X1train[:,1], X1train @ w_sgd.T,'r--',label='Model fit SGD',lw=1)
    plt.xlabel('Floor Size (Square Feet)')
    plt.ylabel('House Price (Dollar)')
    plt.title('Univariate Regression Batch Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/univ_BGD.png')
    plt.show()
    plt.close()




def plot_cost_history(X1, t, epochs,step, learning_rate):
    plt.style.use('ggplot')

    # given epochs
    epochs_lst = np.arange(0, epochs+step, step)
    costs_bgd = [compute_cost(X1, t, (train_BGD(X1, t, learning_rate, epoch))) for epoch in epochs_lst]
    costs_sgd = [compute_cost(X1, t, (train_SGD(X1, t, learning_rate, epoch))) for epoch in epochs_lst]
    #
    # print("costs_bgd = {}".format(costs_bgd))
    # print("costs_sgd = {}".format(costs_sgd))

    # minimum value
    # min_idx_bgd = np.argmin(costs_bgd)
    # min_idx_sgd = np.argmin(costs_sgd)
    # print("np.min(costs_bgd) = {:.5e}".format(np.min(costs_bgd)))
    # print("np.min(costs_sgd) = {:.5e}".format(np.min(costs_sgd)))
    # print("epochs_lst_bgd[min_idx_bgd] = {}".format(epochs_lst[min_idx_bgd]))
    # print("epochs_lst_bgd[min_idx_sgd] = {}".format(epochs_lst[min_idx_sgd]))

    # plot
    plt.plot(epochs_lst, costs_bgd,'bo',label='cost history GD')
    plt.plot(epochs_lst, costs_sgd,'r>',label='cost history SGD',ms=2)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.title('Univariate Cost history')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/univ_cost_history_BGD.png')
    plt.show()
    plt.close()

def print_train_outputs(epochs, learning_rate, mean_train,std_train,w,rmse,cost):
    """Note: in vectorized method w[1] is replaced by w[0][1] and so on"""
    print("               Train Data")
    print("epochs  learning_rate  mean        std       w0          w1          rmse        cost")
    print("{}     {}            {}     {:,.2f}    {:,.2f}  {:,.2f}   {:,.2f}   {:,.2f}".format(
        epochs, learning_rate, mean_train[0][0], std_train[0][0], w[0][0], w[0][1], rmse, cost))



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

def create_gif(X1train, ttrain, X1test,ttest, w, epochs,cost,learning_rate):
    """We create 70 png files and use ImageMagick command to create gif.

    ```convert -loop 0 -delay 100 test/test*.png cost_history.gif```

    In this case Cost function J does not decreases after 70 epochs, so I
    create only 70 png files.

    """
    for epochs in np.arange(70):
        w = train_BGD(X1train, ttrain, learning_rate, epochs)
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


    # Hyperparameters
    epochs, step, learning_rate = 200, 10, 0.1


    # Get w from Train and use it on Test
    w = train_BGD(X1train, ttrain, learning_rate, epochs)
    print('epochs = {} w_BGD = {}'.format(epochs,w))

    # Parameters for Test
    rmse = compute_rmse(X1train, ttrain, w)
    cost = compute_cost(X1train, ttrain, w)
    grad = compute_gradient(X1train, ttrain, w)
    print_train_outputs(epochs, learning_rate, mean_train,std_train,w,rmse,cost)

    # Compare w from normal eqn and BGD
    w_norm_eqn = train_norm_eqn(X1train, ttrain)
    w_bgd = train_BGD(X1train, ttrain, learning_rate, epochs)
    w_sgd = train_SGD(X1train, ttrain, learning_rate, 115)
    print("\n")
    print("w_norm_eqn         = {}".format(w_norm_eqn))
    print("epochs = 200 w_bgd = {}".format(w_bgd))
    print("epochs = 115 w_sgd = {}".format(w_sgd))


    # plots
    plot_cost_history(X1train, ttrain, epochs,step, learning_rate)
    plot_train_test(X1train,ttrain, X1test,ttest,w_bgd, w_sgd)

    # Compare w from BGD and SGD
    #
    w_norm_eqn = train_norm_eqn(X1train, ttrain)
    iters = 200
    a_mins = []
    for i in np.arange(iters):
        w_sgd = train_SGD(X1train, ttrain, learning_rate, i)
        print("\n")
        print("i = {}".format(i))
        print("w_norm_eqn        = {}".format(w_norm_eqn))
        print('iters = {} w_BGD = {}'.format(iters,w))
        print("iters = {} w_SGD = {}".format(i, w_sgd))
        abs_diff_min = [ abs(w_sgd[0][0]-w_norm_eqn[0][0]) + abs(w_sgd[0][1]-w_norm_eqn[0][1]) ][0]
        a_mins.append(abs_diff_min)
        print("abs_diff_min = {}".format(abs_diff_min))
        print('np.argmin(abs_diff_min) = ', np.argmin(a_mins))
        print("len(a_mins) = {}".format(len(a_mins)))


    # Extra
    # create gifs
    # create_gif(X1train, ttrain, X1test,ttest, w, epochs,cost,learning_rate)


if __name__ == "__main__":
    main()
