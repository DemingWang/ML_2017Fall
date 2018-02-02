#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm.

@author: Bhishan Poudel

@date:  Oct 31, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os, shutil
np.random.seed(100)

from sklearn import datasets

def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:,1:]
    Y = data[:,1]

    return X, Y

def plot_contour(X,Y,w,mesh_stepsize):
    """X is without bias. """
    try:
        plt.style.use('seaborn-darkgrid')
        # plt.style.use('ggplot')
        #plt.style.available
    except:
        pass
    # Get data for two classes
    idxN = np.where(np.array(Y)==-1)
    idxP = np.where(np.array(Y)==1)
    XN = X[idxN]
    XP = X[idxP]

    # plot two classes with + and - sign
    fig, ax = plt.subplots()
    ax.set_title('Perceptron Algorithm')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(XN[:,0],XN[:,1],'b_', markersize=8, label="Negative class")
    plt.plot(XP[:,0],XP[:,1],'y+', markersize=8, label="Positive class")
    plt.legend()

    # create a mesh for contour plot
    # We first make a meshgrid (rectangle full of pts) from xmin to xmax and ymin to ymax.
    # We then predict the label for each grid point and color it.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Get 2D array for grid axes xx and yy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize),
                         np.arange(y_min, y_max, mesh_stepsize))

    # Get 1d array for x and y axes
    xxr = xx.ravel()
    yyr = yy.ravel()

    # ones vector
    ones = np.ones(len(xxr))

    # Predict the score
    Xvals  = np.c_[ones, xxr, yyr]
    scores = np.where(np.dot(Xvals,w) >= 0.0, 1, -1)

    # Plot contour plot
    scores = scores.reshape(xx.shape)
    ax.contourf(xx, yy, scores, cmap='Paired')

    # show the plot
    plt.savefig("Perceptron.png")
    plt.show()
    plt.close()

def plot_boundary(X,Y,w):
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass
    
    # plot without legends (quick and dirty)
    # plt.scatter(X[:,0],X[:,1], c=Y, marker = 'o', alpha = 0.8,s=40)
    
    # plot with labels
    idxN = np.where(np.array(Y)==-1)
    idxP = np.where(np.array(Y)==1)
    XN = X[idxN]
    XP = X[idxP]
    plt.scatter(XP[:,1],XP[:,2],c='r', marker='+', label="Positive class",alpha=0.3,s=40)
    plt.scatter(XN[:,1],XN[:,2],c='b', marker='_', label="Negative class",alpha=0.3,s=40)
    plt.title("Perceptron Algorithm")
    plt.xlabel('X0')
    plt.ylabel('X1')
    
    # bias and weight
    b, w1, w0 = w
    
    # slope and y-intercept
    slope  = -w0 / w1
    y_intr = -b  / w1
    
    # plot values
    ymin, ymax = plt.ylim()
    xx = np.linspace(ymin, ymax)
    yy = slope * xx + y_intr
    
    plt.plot(yy,xx, 'k-')
    plt.legend(loc=1)
    plt.show()
    plt.close()

def perceptron_train(data, labels, epochs):
    w = np.zeros(data.shape[1])
    for j in range(epochs):
        errors = 0
        for i in range (data.shape[0]):
            h = np.sign(data[i,:].dot(w))
            if (h!=labels[i]):
                w += labels[i] * data[i,:]
                errors += 1
        if errors == 0:
            print("converged after %s epochs:" %j)
            break
    return w

def perceptron_sgd(X, Y,epochs):
    """
    X: data matrix with bias.
    Y: target
    """
    # add bias to X's first column
    # ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    # X1 = np.append(ones, X, axis=1)


    w = np.zeros(X.shape[1])
    final_iter = epochs

    for epoch in range(epochs):
        print("\n")
        print("epoch: {} {}".format(epoch, '-'*30))

        misclassified = 0
        for i, x in enumerate(X):
            y = Y[i]
            h = np.dot(x, w)*y

            if h <= 0:
                w = w + x*y
                misclassified += 1
                # print('misclassified? yes  w: {} '.format(w,i))

            else:
                # print('misclassified? no  w: {}'.format(w))
                pass

        if misclassified == 0:
            final_iter = epoch
            break

    return w, final_iter

def save_iris_data():
    # get iris data
    iris = datasets.load_iris()
    X = iris.data[0:100, :2]
    Y = iris.target[0:100]
    
    # shuffle data
    np.random.seed(100)
    perm_idx = np.random.permutation(X.shape[0])
    X = X[perm_idx]
    Y = Y[perm_idx]    
    
    # write iris data to a file
    with open('iris.txt','w') as fo:
        i = 0
        for x1,x2 in zip(X[:,0],X[:,1]):
            y = Y[i]
            if y == 0:
                    y = -1
            print(y,' ', x1,' ',x2, file=fo)
            i += 1
    

def main():
    """Run main function."""

    # get iris data
    iris = datasets.load_iris()
    X = iris.data[0:100, :2]
    Y = iris.target[0:100]

    # shuffle data
    np.random.seed(100)
    perm_idx = np.random.permutation(X.shape[0])
    X = X[perm_idx]
    Y = Y[perm_idx]

    # train and test
    Xtr = X[0:80,:]
    Ytr = Y[0:80]

    Xts = X[80:100,:]
    Yts = Y[80:100]

    Xts_ = X[80:100,:]
    Yts_ = Y[80:100]

    Xtr = np.append(np.ones((Xtr.shape[0],1)), Xtr, axis = 1)
    Ytr[Ytr==0] = -1

    Xts = np.append(np.ones((Xts.shape[0],1)), Xts, axis = 1)
    Yts[Yts==0] = -1

    print("Xtr.shape = {}".format(Xtr.shape))
    print("Xtr[0] = {}".format(Xtr[0]))

    # perceptron algorithm
    w = perceptron_train(Xtr,Ytr,10000)
    # w, final_iter = perceptron_sgd(Xtr, Ytr,10000)


    y_pred = np.where(np.dot(Xts,w) >= 0.0, 1, -1)

    # plot boundary
    # plot_contour(Xts_,Yts_,w,mesh_stepsize=0.01)
    
    # boundary plot
    plot_boundary(Xts,Yts,w)
    
    save_iris_data()

if __name__ == "__main__":
    main()
