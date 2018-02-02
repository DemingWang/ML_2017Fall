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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Compute cost for linear regression
# .T is transpose operator
def computeCost(X, y, theta):
    m = len(y)  # number of tranning examples
    J = 0
    J = 1/(2*m) * ((X*theta) - y).T * ((X*theta) - y)
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        theta = theta - (alpha/m)*X.T*(X*theta - y)
        J_history[iter] = computeCost(X, y, theta)
    return [theta, J_history]




def myplot(data):
    print ("Plotting data")
    X = data[:,0]
    y = np.matrix(data[:,1]).T
    m = len(y)

    plt.figure()
    plt.plot(X, y, 'o')
    #plt.savefig('temp.png')
    plt.show()

def grad_desc(data):
    y = np.matrix(data[:,1]).T
    m = len(y)

    print("Running gradient descent.")

    X = np.matrix([np.ones(m), data[:,0]]).T # add a column of ones to X
    theta = np.matrix(np.zeros(2)).T #initialize fitting parameters

    # gradient descent setting
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    computeCost(X, y, theta)

    # Run gradietn descent
    theta, J_hisotry = gradientDescent(X, y, theta, alpha, iterations)

    # Print theta to screen
    print(("Theta found by gradient descent: ", theta[0], theta[1]))

    plt.figure()
    plt.plot(X[:,1], y, 'o', label = 'Training data', color = 'blue')
    plt.plot(X[:,1], X*theta, '-', label = 'Linear regression', color = 'red')
    plt.legend(loc = 4)
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = [1, 3.5]*theta
    print(('For population = 35,000, we predict a profit of ', predict1*1000))
    predict2 = [1, 7]*theta
    print(("For population = 70,000, we predict a profit of", predict2*10000))


def plot_J(data):
    y = np.matrix(data[:,1]).T
    m = len(y)
    X = np.matrix([np.ones(m), data[:,0]]).T # add a column of ones to X


    # Visualizing J(theta_0, theta_1)
    print("Visualizing J(theta_0, theta_1)")

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.matrix([theta0_vals[i], theta1_vals[j]]).T
            J_vals[i,j] = computeCost(X, y, t)

    # transpose J_vals
    J_vals = J_vals.T

    # surface plot
    fig = plt.figure()
    ax = Axes3D(fig)

    # ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_zlabel('J Values')
    plt.show()


    plt.figure(1)
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.show()

def main():
    """Run main function."""
    data = np.loadtxt('ex1data1.txt', delimiter = ',')
    # myplot(data)
    # grad_desc(data)
    plot_J(data)


if __name__ == "__main__":
    main()
