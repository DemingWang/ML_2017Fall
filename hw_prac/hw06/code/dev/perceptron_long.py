#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 7, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import collections
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

##===================Vanilla Perceptron================================
def perceptron_train(data, labels, epochs,verbose=False):
    X = data
    Y = labels
    
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X = np.append(ones, X, axis=1)
    
    w = np.zeros(X.shape[1])
    final_iter = epochs
    mistakes = 0
    
    # debug
    # print("X.shape = {}".format(X.shape)) # (8, 4)
    # print("w.shape = {}".format(w.shape)) # (4,)
    # print("Y.shape = {}".format(Y.shape)) # (8,)

    
    for epoch in range(epochs):
        if verbose:
            print("\n")
            print("epoch: {} {}".format(epoch, '-'*40))
        
        
        misclassified = 0
        for i, x in enumerate(X):
            y = Y[i]
            h = np.dot(x, w)*y

            if h <= 0:
                w = w + x*y
                misclassified += 1
                mistakes += 1
                if verbose:
                    print('{}-{}: misclassified? y  w: {} '.format(epoch,i, w))
                
            else:
                if verbose:
                    print('{}-{}: misclassified? n  w: {}'.format(epoch,i, w))


        if misclassified == 0:
            final_iter = epoch
            print("\nPerceptron converged after: {} iterations".format(final_iter))
            break
    
    if misclassified != 0:
        print("\nPerceptron DID NOT converge until: {} iterations".format(final_iter))
                
    return w, final_iter, mistakes

def perceptron_test(w, X):
    """
    w.shape = (3,)
    data.shape = (21, 3)
    
    :Output:
        score.shape = 21,
    """
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X = np.append(ones, X, axis=1)
    
    return np.sign(X.dot(w))

##=========================Averaged Perceptron==============================
def aperceptron_train_no_bias(data, labels, epochs):
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
    print("Averaged Perceptron converged after: {} iterations".format(final_iter))
                
    # return w, final_iter
    return wbar/tau, final_iter
def aperceptron_train(data, labels, epochs):
    """data: without bias column
    labels: 1d array
    """
    # data and labels
    X = data
    Y = labels
    
    # initialize weights
    w = u = np.zeros(X.shape[1] )
    b = beta = 0
    
    # counters    
    final_iter = epochs
    c = 1
    mistakes = 0
    
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
                mistakes += 1
                
        # update counter regardless of good or bad classification        
        c = c + 1
        
        # break loop if w converges
        if misclassified == 0:
            final_iter = epoch
            print("Averaged Perceptron converged after: {} iterations".format(final_iter))
            break
        
        
    if misclassified != 0:
        print("\nAveraged Perceptron DID NOT converge until: {} iterations".format(final_iter))
        
    # prints
    # print("final_iter = {}".format(final_iter))
    # print("b, beta, c , (b-beta/c)= {} {} {} {}".format(b, beta, c, (b-beta/c)))
    # print("w, u, (w-u/c) {} {} {}".format(w, u, (w-u/c)) )

                
    # return w and final_iter
    w = w - u/c
    b = np.array([b- beta/c])
    w = np.append(b, w)
    
    return w, final_iter, mistakes

##============== Kernel (Dual) Perceptron============================
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

def kperceptron_train(data,labels,epochs,kernel,verbose=False):
    X = data
    y = labels
    
    # initialize alpha
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples, dtype=np.float64)
    final_iter = epochs

    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(X[i], X[j])

    for epoch in range(epochs):
        misclassified = 0
        for i in range(n_samples):
            if np.sign(np.sum(K[:,i] * alpha * y)) != y[i]:
                alpha[i] += 1.0
                misclassified += 1
                
        
        if verbose:
            print("{}: alpha = {}".format(epoch, alpha))    
        
        # break if conveges
        if misclassified == 0:
            final_iter = epoch
            break
            
    print("\nKernel Perceptron converged after: {} iterations".format(final_iter))
       
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
       'data' is a 2D array, with one example per row.
       'kernel' is the kernel function to be used.
    
    :Output:
        a vector with the predicted labels for the examples in 'data'.
        
    """
    pass

##========================Read example=================================
def read_examples(file_name):
    ldata = np.loadtxt(file_name)
    labels = ldata[:,1]
    data  = ldata[:,1:]
    
    return data, labels


##=========================Read data===================================
def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:,:-1]
    Y = data[:,-1]
       
    return X, Y

##============================Plotting tools============================
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

#================Generate Data ======================================
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


##==============================Main module============================
## Main module
##=======================================================================
def main():
    """Run main function."""

    #===========================================================
    ## qn 1a (hw5 qn5a)
    # X, X1, Y = read_data('../data/ex5/ex5.txt')
    # labels = Y
    # epochs = 20
    
    #===========================================================
    ## Generate linearly separable data
    data = '../data/qn1_data/data.txt'
    data_tr = '../data/qn1_data/data_train.txt'
    data_ts = '../data/qn1_data/data_test.txt'
    data_size = 200
    gen_lin_separable_data(data, data_tr, data_ts,data_size)
    
    #===========================================================
    ## Train and test vanilla perceptron
    # epochs = 20
    # X_train, Y_train = read_data(data_tr)
    # X_test,  Y_test = read_data(data_ts)
    # 
    # w, final_iter = perceptron_train(X_train, Y_train, epochs,verbose=0)
    # score = perceptron_test(w, X_test)
    # 
    # correct = np.sum(score == Y_test)
    # print("Total: {} Correct: {} Accuracy = {} %".format(
    #     len(score), correct, correct/ len(score) * 100))

    #===========================================================
    ## Train and test averaged perceptron
    epochs = 200
    X_train, Y_train = read_data(data_tr)
    X_test,  Y_test = read_data(data_ts)
    
    w, final_iter = aperceptron_train(X_train, Y_train, epochs)
    
    score = perceptron_test(w, X_test)
    correct = np.sum(score == Y_test)
    print("Total: {} Correct: {} Accuracy = {} %".format(
        len(score), correct, correct/ len(score) * 100)) 
    
    #===========================================================
    ## kernel perceptron    
    # kernel = quadratic_kernel
    # alpha, sv, sv_y = kperceptron_train(X_train,Y_train,epochs,kernel,verbose=0)
    # print('alpha = ', alpha)
    # print("len(alpha) = {}".format(len(alpha)))

if __name__ == "__main__":
    main()
