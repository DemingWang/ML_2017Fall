#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 7, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt

##=======================================================================
## Vanilla Perceptron
##=======================================================================
def perceptron_train(data, labels, epochs,verbose=False):
    """
    :Input:
       'data' is a 2D array, with one exampel per row.
       'labels' is a 1D array of labels for the corresponding examples.
        'epochs' is the maximum number of epochs.
    
    :Output:
        the weight vector w.
    """
    X = data
    Y = labels
    w = np.zeros(X.shape[1])
    final_iter = epochs
    
    # debug
    # print("X.shape = {}".format(X.shape)) # (8, 4)
    # print("w.shape = {}".format(w.shape)) # (4,)
    # print("Y.shape = {}".format(Y.shape)) # (8,)

    
    for epoch in range(epochs):
        if verbose:
            print("\n")
            print("epoch: {} {}".format(epoch, '-'*30))
        
        
        misclassified = 0
        for i, x in enumerate(X):
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


        if misclassified == 0:
            final_iter = epoch
            break
                
    return w, final_iter
def perceptron_test(w, data):
    """
    :Input:
       'w' is the weight vector.
       'data' is a 2D array, with one example per row.
    
    :Output:
        a vector with the predicted labels for the examples in 'data'. 
    """
    pass

##=======================================================================
## Averaged Perceptron
##=======================================================================
def aperceptron_train_no_bias(data, labels, epochs):
    """
    :Input:
       'data' is a 2D array, with one exampel per row.
       'labels' is a 1D array of labels for the corresponding examples.
        'epochs' is the maximum number of epochs.
    
    :Output:
        the weight vector w.
    """
    X = data
    Y = labels
    w = np.zeros(X.shape[1])
    wbar = np.zeros(X.shape[1])
    final_iter = epochs
    tau = 1
    
    for epoch in range(epochs):
        
        misclassified = 0
        for i, x in enumerate(X):
            t = Y[i]
            h = np.dot(x, w)*t

            if h <= 0:
                w = w + t*x
                misclassified += 1
                
        wbar = wbar + w
        tau = tau + 1

        if misclassified == 0:
            final_iter = epoch
            break
        
    # prints
    print("tau = {}".format(tau))
    print("final_iter = {}".format(final_iter))
                
    # return w, final_iter
    return wbar/tau, final_iter
def aperceptron_train_bias(data, labels, epochs):
    # data and labels
    X = data
    Y = labels
    
    # initialize weights
    w = u = np.zeros(X.shape[1] )
    b = beta = 0
    
    # counters    
    final_iter = epochs
    c = 1
    
    # main average perceptron algorithm
    for epoch in range(epochs):
        # initialize misclassified
        misclassified = 0
        
        # go through all training examples
        for  x,y in zip(X,Y):
            h = np.dot(x, w)*y

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
            break
        
    # prints
    print("final_iter = {}".format(final_iter))
    print("b, beta, c , (b-beta/c)= {} {} {} {}".format(b, beta, c, (b-beta/c)))
    print("w, u, (w-u/c) {} {} {}".format(w, u, (w-u/c)) )

                
    # return w and final_iter
    w = w - u/c
    b = np.array([b- beta/c])
    w = np.append(b, w)
    
    return w, final_iter

##=======================================================================
## Kernel (Dual) Perceptron
##=======================================================================
def quadratic_kernel(x, y):
    """
    :Input: two examples x and y.
    
    :Output: the quadratic kernel value computed as (1+xTy)^2. 
    
    """
    return ( 1 + np.dot(x,y))**2
def kperceptron_train_bias(data, labels, epochs):    
    # get data and labels
    X = data
    Y = labels
    
    # initialize weights
    alpha = np.zeros(X.shape[0])
    b = 0
    
    # final iter counter
    final_iter = epochs

    # kernel perceptron (Daume Page 144)
    mistaken_xm = []
    mistaken_alpha = []
    for epoch in range(epochs):
        
        misclassified = 0
        for x, y in zip(X,Y):
            a = 5
            if y*a <= 0:
                alpha = alpha + y
                b = b + y
                mistaken_xm.append(x) 
                mistaken_alpha.append(alpha) 
            #end if
        # end for
    # end for
    print("mistaken_alpha[0] = {}".format(mistaken_alpha[0]))
    print("len(mistaken_alpha) = {}".format(len(mistaken_alpha)))
    
    return alpha, final_iter

def kperceptron_train(data,labels,epochs,kernel):
    X = data
    y = labels
    T = epochs
    
    # initialize alpha
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples, dtype=np.float64)

    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(X[i], X[j])

    for t in range(T):
        for i in range(n_samples):
            if np.sign(np.sum(K[:,i] * alpha * y)) != y[i]:
                alpha[i] += 1.0
    
    # Get support vectors
    idx = alpha > 1e-5
    alpha = alpha[idx]
    sv = X[idx]
    sv_y = y[idx]
    
    return alpha,sv,sv_y

def kperceptron_test(alpha, data, kernel):
    """ 
    :Input:
       'alpha' is the parameter vector.
       'data' is a 2D array, with one exampel per row.
       'kernel' is the kernel function to be used.
    
    :Output:
        a vector with the predicted labels for the examples in 'data'.
        
    """
    pass

##=======================================================================
## Read example
##=======================================================================

def read_examples(file_name):
    """
    :Input:
       'file_name' is the name of the file containing a set of examples in the 
    sparse feature vector format.
    
    :Output:
       a tuple '(data, labels)' where the 'data' is a two dimensional array 
       containing all feature vectors, one per row, in the same order as in the 
       input file, and the 'labels' is a vector containing the 
       corresponding labels. 
    """
    pass

##=======================================================================
## Read data
##=======================================================================
def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:,:-1]
    Y = data[:,-1]
    
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X1 = np.append(ones, X, axis=1)
    
    # X is needed for plot    
    return X, X1, Y

##=======================================================================
## Plotting tools
##=======================================================================
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


##=======================================================================
## Main module
##=======================================================================
def main():
    """Run main function."""

    # data
    # X, Y = read_data('../data/ex5/ex5.txt')
    X, X1, Y = read_data('../data/practice/data.txt')
    labels = Y
    epochs = 20
    
    # vanilla perceptron
    # w, final_iter = perceptron_train(X1, labels, epochs,verbose=0) # [-13.   2.   3.]
    
    # averaged perceptron
    # w, final_iter = aperceptron_train_no_bias(X, Y, epochs)
    # w, final_iter = aperceptron_train_bias(X, Y, epochs) 
    
    # kernel perceptron    
    kernel = quadratic_kernel
    alpha = kperceptron_train(X,Y,epochs,kernel)
    print('alpha[0] = ', alpha[0])
    print("len(alpha) = {}".format(len(alpha)))
    

if __name__ == "__main__":
    main()
