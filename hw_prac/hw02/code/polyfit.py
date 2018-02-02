#!python
# -*- coding: utf-8 -*-#
"""
Polynomial Regresssion with Ridge Regression and Batch Gradient Descent.

@author: Bhishan Poudel, Physics PhD Student, Ohio University

@date: Oct 2, 2017

@email: bhishanpdl@gmail.com

:Outputs:

  - ../images/hw02qn4b.png
  - ../images/cost_epochs_good_lr.png
  - ../images/cost_epochs_bad_lr.png
  - ../images/cost_history_bgd_unreg.png
  - ../images/cost_history_bgd_reg.png

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
import warnings
warnings.filterwarnings('ignore')  # XXX : Never Recommended, just to ignore plot warnings.
np.set_printoptions(formatter={'float': lambda x: "{:,.4f} ".format(x)})



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

def train_norm_eqn(X, t):
    """Train the data using normal equations.

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
    w = w.reshape(1, X.shape[1])
    print("w.shape normal eqn = {}".format(w.shape)) # 6,1

    return w

def train_ridge_norm_eqn(X, t, shrinkage, M):
    """Train data with ridge regression using normal equations.

    Args:

      X (array): Design matrix of size (m+1, n). I.e. There are
        m features and one bias column in the matrix X.

      t (column): Target column vector.

      shrinkage (float): The shrinkage hyperparameter  for the regularization.

      M (int): Degree of the polynomial to fit.

    Return:

      w(row): Weight vector in the shape of row.

    """
    # First get the identity matrix of size deg+1 by deg+1
    N = len(t)
    I = np.eye(M + 1)
    I[0][0] = 0 # don't regularize bias term.


    # weight for ridge regression from Normal Equations
    w = inv(shrinkage * N * I + X.T @ X )   @ X.T @ t

    return w.reshape(1, X.shape[1])



def train_ridge_BGD(X, t, shrinkage, iters, learning_rate):
    """Calculate weight vector using Ridge Regression L2 norm using Batch Grad Desc.

    .. note::

       Note that X and t should be normalized before running batch grad descent.

    Args:
      X(matrix): Nomalized Design matrix with bias term.

      t(column vector): Normalized Target column vector (shape = 1, samples)

      shrikage(float): L2 regularization shrikage hyper parameter.

      iters(int): Number of iterations.

      learning_rate(float): Learning rate for gradient descent algorithm.

      Return:

        w(row): Weight vector in the shape of row.
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

def train_ridge_BGD_threshold(X, t, shrinkage, threshold, learning_rate, stepsize):
    """Calculate weight vector using Ridge Regression L2 norm using Batch Grad Desc.

    .. note::

       Note that X and t should be normalized before running batch grad descent.

    Args:
      X(matrix): Nomalized Design matrix with bias term.

      t(column vector): Normalized Target column vector (shape = 1, samples)

      shrikage(float): L2 regularization shrikage hyper parameter.

      ratio(float): Ratio of now to previous cost in grad descent calculation.

      learning_rate(float): Learning rate for gradient descent algorithm.

      stepsize(int): Step size of iterations to run. e.g 1 means 1,2,3,4...
      The answer heavily depends on stepsize in SGD.

      Return:

        w(row): Weight vector in the shape of row.

        final_iter(int): Final iteration when the model converges.

        J_hist(list): List of cost histories.

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
    iters = 3000000000000 # Take large number until you break
    final_iter = 0
    J_hist = []
    for i in range(0, iters,stepsize):
        w = train_ridge_BGD(X, t, shrinkage, i, learning_rate)
        J = compute_cost_ridge(X, t, shrinkage, w)
        J_hist.append(J)
        print("i = {:,}".format(i))
        # print("J_prev-J  = {:.5f}    J_prev = {:.5f}     J = {:.5f}  ".format(J_prev-J, J_prev, J ))

        final_iter = i
        if abs(J_prev - J) <= threshold:
            break

        # Update J after the if statement.
        J_prev = J

    final_iter = final_iter + 1
    w = w.reshape(1, X.shape[1])
    return w, final_iter, J_hist


def train_ridge_SGD(X, t, shrinkage, iters, learning_rate):
    """Calculate weight vector using Ridge Regression L2 norm using Batch Grad Desc.

    .. note::

       Note that X and t should be normalized before running batch grad descent.

    Args:
      X(matrix): Nomalized Design matrix with bias term.

      t(column vector): Normalized Target column vector (shape = 1, samples)

      shrikage(float): L2 regularization shrikage hyper parameter.

      iters(int): Number of iterations.

      learning_rate(float): Learning rate for gradient descent algorithm.

      Return:

        w(row): Weight vector in the shape of row.
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

        # shuffle the data for stochastic grad desc.
        np.random.seed(100)
        perm_idx = np.random.permutation(X.shape[0])
        X = X[perm_idx]
        t = t[perm_idx]

        # iterate over each row of data
        for j in range(X.shape[0]):
            # row of data
            Xj = X[j,:]
            hj = Xj @ w.T
            tj = t[j]

            # reshape
            Xj = Xj.reshape(X.shape[1], 1)
            tj = tj.reshape(1,1)
            hj = hj.reshape(1,1)

            # find gradient and update w row by row of design matrix.
            grad_ols =  (hj-tj) @ Xj.T / N
            grad_ridge = (grad_ols + shrinkage  * w )
            w = w - learning_rate * grad_ridge

    # debug
    # print("Xj.shape = {}".format(Xj.shape)) # 6,1
    # print("w.shape = {}".format(w.shape))   # 1,6
    # print("hj.shape = {}".format(hj.shape)) # 1,1
    # print("tj.shape = {}".format(tj.shape)) # 1,1
    # print("Xj = {}".format(Xj))
    # print("tj = {}".format(tj))

    return w.reshape(1, X.shape[1])

def plot_J_hist_threshold(final_iter, J_hist,ofile,title):
    # matplotlib customization
    plt.style.use('ggplot')
    plt.figure(figsize=(12,8))

    plt.plot(range(final_iter), J_hist, label='Cost history')

    plt.xlabel('epochs/stepsize')
    plt.ylabel('Cost  J(w)')
    plt.title(title)
    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ofile)
    plt.show()
    plt.close()

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
    plt.show()
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
    plt.show()
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
    plt.show()
    plt.close()

def normalize_data(fh_train,deg):

    # Get vandermonde matrix with bias column
    X, t = read_data_vander(fh_train, deg)

    # Zscore normalize all the columns of X except 1st bias column.
    Xnot0 = X[:, 1:]
    Xnot0normalized = (Xnot0 - np.mean(Xnot0, axis=0,keepdims=True)) / np.std(Xnot0, axis=0, keepdims=True)
    # print("Xnot0[0] = {}".format(Xnot0[0]))
    # print("Xnot0normalized[0] = {}".format(Xnot0normalized[0]))

    # Append bias column back to zcale normalized matrix X.
    X = np.append(np.ones_like(t), Xnot0normalized, axis=1)

    # We do not feature normalize the target vector.
    # t = (t - np.mean(t, axis=0,keepdims=True)) / np.std(t, axis=0, keepdims=True)
    # print("X[0] = {}".format(X[0]))
    # print("t.shape = {}".format(t.shape))
    return X, t

def choose_learning_rate(X,t):
    shrinkage, deg, epochs = 0.0 , 5, np.arange(0,510,10)
    learning_rates = [10**i for i in range(-4,2)]
    Jvals_lst = [ [compute_cost_ridge(X, t, shrinkage,  train_ridge_BGD(X, t, shrinkage, epoch, learning_rate))
             for epoch in epochs] for learning_rate in learning_rates]
    plot_cost_epoch(Jvals_lst, epochs)

def plot_cost_hist_unreg(X,t):
    shrinkage, threshold, learning_rate, stepsize = 0, 1e-4, 0.1, 1
    w_unreg_bgd, final_iter, J_hist = train_ridge_BGD_threshold(X, t, shrinkage, threshold, learning_rate, stepsize)
    print("w_unreg_bgd = {}".format(w_unreg_bgd))
    print("final_iter = {}".format(final_iter))
    print("len(J_hist) = {}".format(len(J_hist)))
    ofile = '../images/cost_history_bgd_unreg.png'
    title = 'Cost history GD Unregularized'
    plot_J_hist_threshold(final_iter, J_hist, ofile,title)

def compare_w_unreg(X,t):

    # run sgd fitting model
    shrinkage, threshold, learning_rate, stepsize = 0, 1e-4, 0.1, 1
    w_unreg_bgd, final_iter, J_hist = train_ridge_BGD_threshold(X, t, shrinkage, threshold, learning_rate, stepsize)


    # After fitting compare normal, gd and sgd methods.
    shrinkage, iters, learning_rate, deg = 0, final_iter, 0.1, 5
    w_norm_eqn = train_ridge_norm_eqn(X, t, shrinkage, deg)
    w_unreg_bgd      = train_ridge_BGD(X, t, shrinkage, iters, learning_rate)
    w_unreg_sgd      = train_ridge_SGD(X, t, shrinkage, iters, learning_rate)
    print("shrinkage = {:.2f} final_iter = {:d} learning_rate = {:.2f} deg = {:d} threshold = {:.2e}".format(
        shrinkage, final_iter, learning_rate, deg, threshold))
    print("w_norm_eqn  = {}".format(w_norm_eqn))
    print("w_unreg_bgd = {}".format(w_unreg_bgd))
    print("w_unreg_sgd = {}".format(w_unreg_sgd))

def plot_cost_hist_reg(X,t):
    shrinkage, threshold, learning_rate, stepsize = 0.1, 1e-10, 0.1, 1
    w_reg_bgd, final_iter, J_hist = train_ridge_BGD_threshold(X, t, shrinkage, threshold, learning_rate, stepsize)
    # print("w_unreg_bgd = {}".format(w_reg_bgd))
    # print("final_iter = {}".format(final_iter))
    # print("len(J_hist) = {}".format(len(J_hist)))
    ofile = '../images/cost_history_bgd_reg.png'
    title = 'Cost history GD Regularized'
    plot_J_hist_threshold(final_iter, J_hist, ofile,title)

def compare_w_reg(X,t):
    shrinkage, threshold, learning_rate, stepsize, deg = 0.1, 1e-10, 0.1, 1, 5
    w_reg_bgd, final_iter, J_hist = train_ridge_BGD_threshold(X, t, shrinkage, threshold, learning_rate, stepsize)
    w_norm_eqn = train_ridge_norm_eqn(X, t, shrinkage, deg)
    w_reg_bgd      = train_ridge_BGD(X, t, shrinkage, final_iter, learning_rate)
    w_reg_sgd      = train_ridge_SGD(X, t, shrinkage, final_iter, learning_rate)
    print("shrinkage  = {:.2f} final_iter = {:d} learning_rate = {:.2f} deg = {:d} threshold = {:.2e}".format(
        shrinkage, final_iter, learning_rate, deg, threshold))
    print("w_norm_eqn = {}".format(w_norm_eqn))
    print("w_reg_bgd  = {}".format(w_reg_bgd))
    print("w_reg_sgd  = {}".format(w_reg_sgd))

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

    X, t = normalize_data(fh_train,deg=5)

    # qn4 a,b,c
    plot_3data()


    # qn4d 1 without regularization
    choose_learning_rate(X,t)
    plot_cost_hist_unreg(X,t)
    compare_w_unreg(X,t)

    # qn4d 2 with regularization
    plot_cost_hist_reg(X,t)
    # compare_w_reg(X,t)



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
    print("\n\nBegin time: ", begin_ctime)
    print("End   time: ", end_ctime, "\n")
    print("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
