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

def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:,:-1]
    Y = data[:,-1]
    
    return X, Y

def plot_boundary(X,Y,w,epoch):
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
           
    # plot two classes
    plt.scatter(XN[:,0],XN[:,1],c='b', marker='_', label="Negative class")
    plt.scatter(XP[:,0],XP[:,1],c='r', marker='+', label="Positive class")
    # plt.plot(XN[:,0],XN[:,1],'b_', markersize=8, label="Negative class")
    # plt.plot(XP[:,0],XP[:,1],'r+', markersize=8, label="Positive class")
    plt.title("Perceptron Algorithm iteration: {}".format(epoch))
    
    # plot decision boundary orthogonal to w
    # w is w2,w1, w0  last term is bias.
    if len(w) == 3:
        a  = -w[0] / w[1]
        b  = -w[0] / w[2]
        xx = [ 0, a]
        yy = [b, 0]
        plt.plot(xx,yy,'--g',label='Decision Boundary')

    if len(w) == 2:
        x2=[ w[0],  w[1],  -w[1],  w[0]]
        x3=[ w[0],  w[1],   w[1], -w[0]]

        x2x3 =np.array([x2,x3])
        XX,YY,U,V = list(zip(*x2x3))
        ax = plt.gca()
        ax.quiver(XX,YY,U,V,scale=1, color='g')
    
    # Add labels
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    
    # lines from origin
    plt.axhline(y=0, color='k', linestyle='--',alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='--',alpha=0.2)
    plt.grid(True)
    plt.legend(loc=1)
    plt.show()
    plt.savefig('img/iter_{:03d}'.format(int(epoch)))
    
    # Always clost the plot
    plt.close()

def predict(X,w):
    return np.sign(np.dot(X, w))

def plot_contour(X,Y,w,mesh_stepsize):
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
    
    # Get 2D array for grid axes xx and yy  (shape = 700, 1000)
    # xx has 700 rows.
    # xx[0] has 1000 values.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize),
                         np.arange(y_min, y_max, mesh_stepsize))
    
    # Get 1d array for x and y axes
    xxr = xx.ravel()  # shape (700000,)
    yyr = yy.ravel()  # shape (700000,)
    
    # ones vector
    # ones = np.ones(xxr.shape[0]) # shape (700000,)
    ones = np.ones(len(xxr)) # shape (700000,)
    
    # Predict the score
    Xvals  = np.c_[ones, xxr, yyr]
    scores = predict(Xvals, w)

    # Plot contour plot
    scores = scores.reshape(xx.shape)
    ax.contourf(xx, yy, scores, cmap=plt.cm.Paired)
    # print("xx.shape = {}".format(xx.shape))               # (700, 1000)
    # print("scores.shape = {}".format(scores.shape))       # (700, 1000)
    # print("scores[0].shape = {}".format(scores[0].shape)) # (1000,)
    
    # show the plot
    plt.savefig("Perceptron.png")
    plt.show()
    plt.close()

def aperceptron_sgd(X, Y,epochs):    
    # initialize weights
    w = np.zeros(X.shape[1] )
    u = np.zeros(X.shape[1] )
    b = 0
    beta = 0
    
    # counters    
    final_iter = epochs
    c = 1
    converged = False
    
    # main average perceptron algorithm
    for epoch in range(epochs):
        # initialize misclassified
        misclassified = 0
        
        # go through all training examples
        for  x,y in zip(X,Y):
            h = y * (np.dot(x, w) + b)

            if h <= 0:
                w = w + y*x
                b = b + y
                
                u = u+ y*c*x
                beta = beta + y*c
                misclassified += 1
                
        # update counter regardless of good or bad classification        
        c = c + 1
        
        # break loop if w converges
        if misclassified == 0:
            final_iter = epoch
            converged = True
            print("Averaged Perceptron converged after: {} iterations".format(final_iter))
            break
    
    if converged == False:
        print("Averaged Perceptron DID NOT converged.")
            
    # prints
    # print("final_iter = {}".format(final_iter))
    # print("b, beta, c , (b-beta/c)= {} {} {} {}".format(b, beta, c, (b-beta/c)))
    # print("w, u, (w-u/c) {} {} {}".format(w, u, (w-u/c)) )

                
    # return w and final_iter
    w = w - u/c
    b = np.array([b- beta/c])
    w = np.append(b, w)
    
    return w, final_iter

def main():
    """Run main function."""

    X, Y = read_data('data.txt') # X is without bias
    max_iter = 20
    w, final_iter = aperceptron_sgd(X,Y,max_iter)
    print('w = ', w)
    
    plot_boundary(X,Y,w,final_iter)
    
    # contour plot
    mesh_stepsize = 0.01
    plot_contour(X,Y,w,mesh_stepsize)

if __name__ == "__main__":
    main()
