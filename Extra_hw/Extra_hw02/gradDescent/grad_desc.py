#!python
# -*- coding: utf-8 -*-#
"""
:Title: Gradient Descent.

@author: Bhishan Poudel

@date: Sep 24, 2017

@email: bhishanpdl@gmail.com

The cost function is given by

.. math::

  J(w) = \\frac{1}{2N} \sum_{n=1}^N (h(x_n,w) - t_n)^2

Minimizing the cost function w.r.t. w gives two system of liner equations:



We solve these normal equations and find the values w0 and w1.
"""
# Imports
import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
import numpy.polynomial.polynomial as poly

# checking
# import statsmodels.api as sm # sm 0.8.0 gives FutureWarning




def read_data(infile):
    """Read the datafile and return arrays"""
    data = np.genfromtxt(infile, delimiter=None,dtype=int)
    X = data[:,0].reshape(len(data),1)
    t = data[:,-1].reshape(len(data),1)

    return [X, t]

def cost_function(X, t, w):
    """
    cost_function(X, y, beta) computes the cost of using beta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    n = len(t)

    ## Calculate the cost with the given parameters
    J = np.sum((X.dot(w)-t)**2)/2/n

    return J

# def gd(X, t, w, alpha, iters):
def gd(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.T
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta) - y

        for j in range(len(parameters)):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

def myplot(fh_train,fh_test,w):
    # matplotlib customization
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    # data
    Xtrain, ttrain = read_data(fh_train)
    Xtest, ttest = read_data(fh_test)
    Xhyptest = Xtest * w[1] + w[0]


    # plot with label, title
    ax.scatter(Xtrain,ttrain,color='b',marker='o', label='Univariate Train')
    ax.scatter(Xtest,ttest,c='limegreen', marker='^', label='GradDesc Test')
    ax.plot(Xtest,Xhyptest,'r--',label='Best Fit')

    # set xlabel and ylabel to AxisObject
    ax.set_xlabel('Floor Size (Square Feet)')
    ax.set_ylabel('House Price (Dollar)')
    ax.set_title('GradDesc Regression')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('GradDesc.png')
    plt.show()

##=======================================================================
## Main Program
##=======================================================================
def main():
    """Run main function."""
    parser = argparse.ArgumentParser('GradDesc Exercise.')
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../data/univariate',
                        help='Directory for the univariate houses dataset.')
    FLAGS, unparsed = parser.parse_known_args()

    # Data file paths
    fh_train = FLAGS.input_data_dir + "/train.txt"
    fh_test  = FLAGS.input_data_dir + "/test.txt"

    # Print weight vector
    Xtrain, ttrain = read_data(fh_train)
    w = np.array([0, 0])
    w = w.reshape(1,2)

    (b, c) = gd(Xtrain, ttrain, w, alpha=0.01, iters=100)
    print("(b,c) = {}".format((b,c)))




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
