#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm.

@author: Bhishan Poudel

@date:  Nov 16 , 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os, shutil,subprocess
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
    # plt.show()
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
    plt.plot(XP[:,0],XP[:,1],'y+', markersize=8, label="Positive class")
    plt.plot(XN[:,0],XN[:,1],'b_', markersize=8, label="Negative class")
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

def perceptron_train(X, Y,epochs,makeplot,verbose=False):
    """
    X: data matrix without bias.
    Y: target
    """
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X1 = np.append(ones, X, axis=1)
    
    
    w = np.zeros(X1.shape[1])
    final_iter = epochs
    
    for epoch in range(epochs):
        print("\n")
        print("epoch: {} {}".format(epoch, '-'*30))
        
        if makeplot:
            plot_boundary(X,Y,w,epoch)
        
        misclassified = 0
        for i, x in enumerate(X1):
            y = Y[i]
            h = np.dot(x, w)*y

            if h <= 0:
                w = w + x*y
                misclassified += 1
                if verbose:
                    print('misclassified? yes  w: {} '.format(w,i))
                
            else:
                if verbose:
                    print('misclassified? no  w: {}'.format(w))
                pass

        if misclassified == 0:
            final_iter = epoch
            break
                
    return w, final_iter

def gen_lin_separable_data(data, data_tr, data_ts,data_size):
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, size=int(data_size/2))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, size=int(data_size/2))
    y2 = np.ones(len(X2)) * -1
    
    
    with open(data,'w') as fo, \
         open(data_tr,'w') as fo1, \
         open(data_ts,'w') as fo2:
        for i in range( len(X1)):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X2[i][0], X2[i][1], y2[i])
            fo.write(line)
            fo.write(line2)
        
        for i in range( len(X1) - 20):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X2[i][0], X2[i][1], y2[i])
            fo1.write(line)
            fo1.write(line2)
        
        for i in range((len(X1) - 20), len(X1) ):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X2[i][0], X2[i][1], y2[i])
            fo2.write(line)
            fo2.write(line2)

def create_gif():
    cmd = """
    convert -loop 0 -delay 100 img/*.png perceptron.gif
    open perceptron.gif
    """
    subprocess.call(cmd,shell=True)

def main():
    """Run main function."""

    X_train, Y_train = read_data('../data/extra/data_train.txt') # X is unbiased
    X_test, Y_test = read_data('../data/extra/data_test.txt')    # X is unbiased
    max_iter = 20
    w, final_iter = perceptron_train(X_train,Y_train,max_iter,makeplot=True)
    print('w = ', w)
    
    plot_boundary(X_test,Y_test,w,final_iter)
    
    # contour plot
    mesh_stepsize = 0.01
    plot_contour(X_test,Y_test,w,mesh_stepsize)
    
    # create gif
    create_gif()

if __name__ == "__main__":
    main()
