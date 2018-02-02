#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 13, 2017 Wed
# Last update :
###########################################################################
"""
:Topic: Linear Regression Using Gradient Descent

:Runtime:

"""
# Imports
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

def lin_regr1():
    n = 50
    x = np.random.randn(n)
    y = x * np.random.randn(n)

    fig, ax = plt.subplots()
    fit = np.polyfit(x, y, deg=1)
    ax.plot(x, fit[0] * x + fit[1], color='red')
    ax.scatter(x, y)

    plt.show()

def lin_regr2():
    # sample data
    x = np.arange(10)
    y = 5*x + 10

    # fit with np.polyfit
    m, b = np.polyfit(x, y, 1)

    plt.plot(x, y, '.')
    plt.plot(x, m*x + b, '-')
    plt.show()

def lin_regr3():
    X = np.random.rand(100)
    Y = X + np.random.rand(100)*0.1

    results = sm.OLS(Y,sm.add_constant(X)).fit()

    # print (results.summary())
    print("results.params = ", results.params)
    w = results.params

    plt.scatter(X,Y)

    X_plot = np.linspace(0,1,100)
    plt.plot(X_plot, X_plot*w[1] + w[0])

    plt.show()

def lin_regr4():
    x = np.random.rand(100)
    y = x + np.random.rand(100)*0.1
    plt.scatter(x,y)
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x))
    plt.show()


if __name__ == "__main__":
    lin_regr3()
