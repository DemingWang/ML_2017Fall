#!python
# -*- coding: utf-8 -*-#
"""
:Title: Polynomial Regresssion with Ridge Regression and Batch Gradient Descent.

@author: Bhishan Poudel, Physics PhD Student, Ohio University

@date: Sep 29, 2017

@email: bhishanpdl@gmail.com

The cost function for the Ridge Regression is given by

.. math::

  J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2 + \
  \\frac{\lambda}{2} ||w||^2

In this case we use batch gradient descent method to model the training data.
"""
# Imports
import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv, norm,pinv
from numpy import sum, sqrt, array, log, exp
np.set_printoptions(formatter={'float': lambda x: "{:,.4f} ".format(x)})
import warnings # matplotlib UserWarning for plot lables
warnings.filterwarnings("ignore")

# Read data matrix X and labels t from text file.
def read_data(file_name):
    *X,t = np.genfromtxt(file_name,unpack=True,dtype=np.float64)
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)

    return X, t


def read_data_vander(infile, M):
    """Read the dataset and return vandermonde matrix Xvan for given degree M.
      """
    X, t = np.genfromtxt(infile, delimiter=None, dtype=np.double,unpack=True)
    Xvan = np.vander(X, M + 1, increasing =True)
    t = t.reshape(len(X),1) # make column vector

    # # debug
    # print("Xvan.shape = {}".format(Xvan.shape)) # e.g 20, 10
    # print("t.shape = {}".format(t.shape)) # e.g. 20, 1

    return Xvan, t

# Compute objective function (cost) on dataset (X, t).
def compute_cost_ridge(X1, t, shrinkage, w):
    """Compute the cost function.

    .. math:: J = \\frac{1}{2N} \sum_{i=1}^{N} (h_n - t_n)^2 + \
    \\frac{\\lambda}{2} ||w||^2

    Args:
      X1(matrix): Design matrix with bias column.
      t(column vector): Target column vector.
      shrikage(float) : Shrinkage hyperparameter for Ridge L2 normalization.
      w(row vector) : Weight row vector.

    Return:
      J(float): Cost value.

    """

    # Compute cost
    N = float(len(t))
    h = X1 @ w.T
    J = np.sum((h - t) ** 2) /2 / N + shrinkage / 2 * np.square(norm(w))

    return J

def train(X, t):
    """Train the data and return the weights w.

    This model uses OLS method to train the data without the penalty term.

    .. math::

      J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2

    Args:

      X (array): Design matrix of size (m+1, n). I.e. There are
        m features and one bias column in the matrix X.

      t (column): target column vector
    """
    w = inv(X.T @ X)  @ X.T @ t   # M = 5

    # make w row vector
    w = w.T
    print("w.shape normal eqn = {}".format(w.shape)) # 6,1

    return w

def train_regularized(Xm1, t, lam, M):
    """Ridge Regularization (L2 normalization) with square penalty term.

    The cost function for ridge regularization is

    .. math::

      J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2 + \\frac{\lambda}{2} ||w||^2

    Minimizing cost function gives the weight vector w.
    Here :math:`\\lambda` is the hyperparameter chosen from validation set
    with lowest rmse for given values of degrees of polynomial. Different may
    give the same minimum rmse and we choose one of them.

    .. math::

      w = (\lambda N I) (X^T t)

    Args:

      Xm1 (array): Design matrix of size (m+1, n). I.e. There are
        m features and one bias column in the matrix X.

      t (column): Target column vector.

      lam (float): The shrinkage hyperparameter  for the regularization.

      M (int): Degree of the polynomial to fit.

    .. note::

       Here the design matrix X should have one extra bias term.
       The function read_data_vander returns X with one extra

    .. warning::

       The operator @ requires python >= 3.5

    """
    # debug
    # Example M = 9, Xm1 has shape 10,10 and t has shape 10,1
    # print("Xm1.shape = {}".format(Xm1.shape))
    # print("t.shape = {}".format(t.shape))


    # First get the identity matrix of size deg+1 by deg+1
    N = len(t)
    I = np.eye(M + 1)
    I[0][0] = 0 # don't regularize bias term.


    # weight for ridge regression from Normal Equations
    w = inv(lam * N * I + Xm1.T @ Xm1 )   @ Xm1.T @ t

    w = w.T
    print("w normal = {}".format(w))
    print("w.shape = {}".format(w.shape))

    return w



def ridge_BGD(X, t, shrinkage, iters, learning_rate):
    """Calculate weight vector using Ridge Regression L2 norm using Batch Grad Desc.

    .. note::

       Note that X and t should be normalized before running batch grad descent.

    Args:
      X(matrix): Nomalized Design matrix with bias term.

      t(column vector): Normalized Target column vector (shape = 1, samples)

      shrikage(float): L2 regularization shrikage hyper parameter.

      iters(int): Number of iterations.

      learning_rate(float): Learning rate for gradient descent algorithm.

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
    for i in range(0, iters):
        h = X @ w.T

        # MSE = np.square(h - t).mean()
        # print("MSE = {}".format(MSE))

        grad_ols =  (h-t).T @ X / N

        # print("grad_ols.shape = {}".format(grad_ols.shape)) # 1,6 w is also 1,6

        grad_ridge = (grad_ols + shrinkage  * w )
        w = w - learning_rate * grad_ridge

    # make w row vector
    w = w.reshape(1, X.shape[1]) # shape = 1, feature + 1
    return w

def ridge_SGD(X, t, shrinkage, iters, learning_rate):
    """Calculate weight vector using Ridge Regression L2 norm using Batch Grad Desc.

    .. note::

       Note that X and t should be normalized before running batch grad descent.

    Args:
      X(matrix): Nomalized Design matrix with bias term.

      t(column vector): Normalized Target column vector (shape = 1, samples)

      shrikage(float): L2 regularization shrikage hyper parameter.

      iters(int): Number of iterations.

      learning_rate(float): Learning rate for gradient descent algorithm.

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
    for i in range(0, iters):
        # shuffle the data
        perm_idx = np.random.permutation(X.shape[0])
        X = X[perm_idx]
        t = t[perm_idx]

        # get hypothesis
        for j in range(X.shape[0]):

            Xj = X[j,:]
            hj = Xj @ w.T
            tj = t[j]

            # reshape
            Xj = Xj.reshape(X.shape[1], 1)
            tj = tj.reshape(1,1)
            hj = hj.reshape(1,1)

            # MSE = np.square(h - t).mean()
        # print("MSE = {}".format(MSE))
            #
            grad_ols =  (hj-tj) @ Xj.T / N
            #
            # # print("grad_ols.shape = {}".format(grad_ols.shape)) # 1,6 w is also 1,6
            #
            grad_ridge = (grad_ols + shrinkage  * w )
            w = w - learning_rate * grad_ridge

    # debug
    # print("Xj.shape = {}".format(Xj.shape)) # 6,1
    # print("w.shape = {}".format(w.shape)) # 1,6
    # print("hj.shape = {}".format(hj.shape)) # 1,1
    # print("tj.shape = {}".format(tj.shape)) # 1,1
    # print("Xj = {}".format(Xj))
    # print("tj = {}".format(tj))

    # make w row vector
    w = w.reshape(1, X.shape[1]) # shape = 1, feature + 1
    return w

def ridge_BGD_diff(X, t, shrinkage, difference, learning_rate=0.1):
    """Calculate weight vector using Ridge Regression L2 norm using Batch Grad Desc.

    .. note::

       Note that X and t should be normalized before running batch grad descent.

    Args:
      X(matrix): Nomalized Design matrix with bias term.

      t(column vector): Normalized Target column vector (shape = 1, samples)

      shrikage(float): L2 regularization shrikage hyper parameter.

      ratio(float): Ratio of now to previous cost in grad descent calculation.

      learning_rate(float): Learning rate for gradient descent algorithm.

    """
    X=np.array(X)
    t = np.array(t)
    t =t.reshape(len(t),1)
    N = len(t)
    w = np.zeros(X.shape[1]) # Initiliaze to zeros.
    w = w.reshape(1,len(w))

    # debug
    # print("\n\n")
    # print("Inside ridge_BGD")
    # print("x.shape = {}".format(X.shape))
    # print("t.shape = {}".format(t.shape))
    # print("w.shape = {}".format(w.shape))
    # print("shrinkage = {}".format(shrinkage))
    # print("iters = {}".format(iters))
    # print("learning_rate = {}".format(learning_rate))
    J_prev = 1e9 # initialize J to large initial value.
    iters = 300 # Take large number until you break
    for i in range(0, iters,10):
        w = ridge_BGD(X, t, shrinkage, i, learning_rate)
        J = compute_cost_ridge(X, t, 0, w)
        print("J = {} J_prev = {} J_prev-J = {}".format(
            J, J_prev, J_prev-J))

        if J_prev - J <= 0.001:
            print("iteration = {}".format(i))
            break

        # Update J after the if checking.
        J_prev = J

    # make w row vector
    w = w.reshape(1, X.shape[1]) # shape = 1, feature + 1
    print('w = ', w)
    return w



def plot_cost_epoch(Jvals_lst, epochs):
    # matplotlib customization
    plt.style.use('ggplot')

    # without lr 1 and 10
    plt.plot(epochs, Jvals_lst[0], label='learning rate = 0.0001')
    plt.plot(epochs, Jvals_lst[1], label='learning rate = 0.001')
    plt.plot(epochs, Jvals_lst[2], label='learning rate = 0.01')
    plt.plot(epochs, Jvals_lst[3], label='learning rate = 0.1')
    plt.xlabel('epoch')
    plt.ylabel('Cost  J(w)')
    plt.title('Choosing hyperparameter learning_rate')
    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/cost_epochs_good_lr.png')
    # plt.show()
    plt.close()

    # For learning rate 1 and 10 we get nans
    plt.plot(epochs, Jvals_lst[4], label='learning rate = 1')
    plt.plot(epochs, Jvals_lst[5], label='learning rate = 10')
    plt.xlabel('epoch')
    plt.ylabel('Cost  J(w)')
    plt.title('Choosing hyperparameter learning_rate')
    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/cost_epochs_bad_lr.png')
    # plt.show()
    plt.close()


def plot_3data():
    data_files = ['train','test','devel']
    styles = ['bo','g^','r>']

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6),dpi=80)
    for i, data_file in enumerate(data_files):
        X, t = read_data('../data/polyfit/{}.txt'.format(data_file))
        ax = plt.subplot(3,1,i+1)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.plot(X,t,styles[i])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('../images/hw02qn4b.png')
    # plt.show()
    plt.close()




##=======================================================================
## Main Program
##=======================================================================
def main():
    """Run main function."""
    parser = argparse.ArgumentParser('Univariate Exercise.')
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../data/polyfit',
                        help='Directory for the polyfit dataset.')
    FLAGS, unparsed = parser.parse_known_args()


    ##=======================================================================
    ## Question 4: Polynomial Univariate Ridge Regularization
    ##=======================================================================
    fh_train = FLAGS.input_data_dir + "/train.txt"
    fh_test = FLAGS.input_data_dir + "/test.txt"
    fh_valid = FLAGS.input_data_dir + "/devel.txt"

    # qn 4a is already done (datafiles are created already)
    # qn 4b plotting 3 data
    plot_3data()

    # qn 4c is theory to find gradient for ridge regression.
    # qn 4d without regularization deg = 5
    #
    learning_rates = [10**i for i in range(-4,2)]
    # shrinkage = 0**-10  # log shrinkage = -10 (given in question)
    shrinkage = 0.0  # for finding lambda choose shrinkage lambda = 0
    deg = 5
    epochs = np.arange(0,210,10)
    # print("epochs = {}".format(epochs))

    # Get vander matrix with bias column
    X, t = read_data_vander(fh_train, deg)

    # Zscale normalize all the columns of X except 1st bias column.
    Xnot0 = X[:, 1:]
    Xnot0normalized = (Xnot0 - np.mean(Xnot0, axis=0,keepdims=True)) / np.std(Xnot0, axis=0, keepdims=True)
    # print("Xnot0[0] = {}".format(Xnot0[0]))
    # print("Xnot0normalized[0] = {}".format(Xnot0normalized[0]))

    # Append bias column back to zcale normalized matrix X.
    X = np.append(np.ones_like(t), Xnot0normalized, axis=1)
    # t = (t - np.mean(t, axis=0,keepdims=True)) / np.std(t, axis=0, keepdims=True)

    # print("X[0] = {}".format(X[0]))
    # print("t.shape = {}".format(t.shape))

    # After normalizing, run batch grad descent
    w = ridge_BGD(X, t, 0, 60000, 0.1)
    J = compute_cost_ridge(X, t, shrinkage, w)
    print("w for BGD = {}".format(w))


    # After normalizing, run sto grad descent
    w = ridge_SGD(X, t, 0, 60000, 0.1)
    print("w for SGD = {}".format(w))



    # Normal equation method
    # w = train(X, t)
    w = train_regularized(X, t, 0, 5)
    print("w for Normal = {}".format(w))





    # print("J = {}".format(J))
    #
    # h = X @ w.T
    # print("h.T = \n", h.T)
    #
    # print("X[0] = {}".format(X[0]))
    # print("t[0] = {}".format(t[0]))
    # print("w = {}".format(w))
    # print("w.shape = {}".format(w.shape))

    # #For choose hyperparameter learning_rate using plot J vs. epochs
    #
    # Jvals_lst = [ [compute_cost_ridge(X, t, shrinkage,  ridge_BGD(X, t, shrinkage, epoch, learning_rate))
    #          for epoch in epochs] for learning_rate in learning_rates]
    #
    # plot_cost_epoch(Jvals_lst, epochs)



    # for learning_rate in learning_rates:
    #     Jvals = [compute_cost_ridge(X, t, shrinkage,  ridge_BGD(X, t, shrinkage, epoch, learning_rate))
    #          for epoch in epochs]
    #
    #     # print("Jvals = {}".format(Jvals))
    #     plot_cost_epoch(Jvals, epochs,learning_rate)


    # # From the plots I found learning_rate = 0.1 is the best. ( 1 and 10 gives nans)
    # learning_rate = 0.1
    # difference = 0.001
    # w = ridge_BGD_diff(X, t, shrinkage, difference, learning_rate)

if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    #  Run the main program
    main()


    # Print the time taken
    program_end_time = time.time()
    end_ctime        = time.ctime()
    seconds          = program_end_time - program_begin_time
    m, s             = divmod(seconds, 60)
    h, m             = divmod(m, 60)
    d, h             = divmod(h, 24)
    print("\nBegin time: ", begin_ctime)
    print("End   time: ", end_ctime, "\n")
    print("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
