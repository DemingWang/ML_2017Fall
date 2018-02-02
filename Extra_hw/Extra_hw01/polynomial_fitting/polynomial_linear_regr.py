#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        :
# Last update :
###########################################################################
"""
:Topic: Polynomial Linear Regression. (Polynomial in X and linear in w)

` <http://blog.mmast.net/least-squares-fitting-numpy-scipy>`_
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy.linalg import inv,norm,lstsq

def create_data():
    np.random.seed(0)
    f = np.poly1d([5, 1])

    x = np.linspace(0, 10, 30)
    y = f(x) + 6*np.random.normal(size=len(x))
    xn = np.linspace(0, 10, 200)

    # plt.plot(x, y, 'or')
    # plt.show()

    return x,y,xn

def fitting(x,y):
    a = np.vstack([x, np.ones(len(x))]).T
    w = inv(a.T @ a) @ (a.T @ y)
    print("w = ", w)

    # fitting using lstsq
    w = lstsq(a,y)[0]
    print("w = ", w)

    # fitting using np.polyfit
    w = np.polyfit(x,y,1)
    print('w = ', w)


    return w

def plot_fit(x,y):
    m, c = np.polyfit(x, y, 1)
    yn = np.polyval([m, c], xn)

    plt.plot(x, y, 'or')
    plt.plot(xn, yn)
    plt.show()



def main():
    """Run main function."""
    x,y,xn = create_data()
    fitting(x,y)

if __name__ == "__main__":
    main()
