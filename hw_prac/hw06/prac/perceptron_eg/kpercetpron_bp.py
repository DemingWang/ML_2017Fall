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


##============== Kernel (Dual) Perceptron============================
def quadratic_kernel(x, y):
    """
    :Input: two examples x and y.
    
    :Output: the quadratic kernel value computed as (1+xTy)^2. 
    
    """
    return ( 1 + np.dot(x,y))**2

def kperceptron_train(data,labels,epochs,kernel,verbose=False):
    X = data
    y = labels
    y = np.array(y)
    y[y==0] = -1
    
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples, dtype=np.float64)

    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(X[i], X[j])

    for t in range(epochs):
        for i in range(n_samples):
            if np.sign(np.sum(K[:,i] * alpha * y)) != y[i]:
                alpha[i] += 1.0
    
    idx = alpha > 1e-5
    alpha = alpha[idx]
    sv = X[idx]
    sv_y = y[idx]
    
    return alpha,sv,sv_y

def kperceptron_project(X,kernel,alpha,sv,sv_y):
    
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, svy, sv_ in zip(alpha, sv_y, sv):
            s += a * svy * 1
            s += a * svy * kernel(X[i], sv_)
        y_predict[i] = s
    return y_predict

def kperceptron_test(X,kernel,alpha,sv, sv_y):
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape

    return np.sign(kperceptron_project(X,kernel,alpha,sv, sv_y))


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
    data_tr = '../../data/extra/data_train.txt'
    data_ts = '../../data/extra/data_test.txt'
    
    #===========================================================
    ## kernel perceptron    
    epochs = 200
    X_train, y_train = read_data(data_tr)
    X_test,  y_test = read_data(data_ts)
    kernel = quadratic_kernel
    alpha, sv, sv_y = kperceptron_train(X_train,y_train,epochs,kernel,verbose=0)
    print('alpha = ', alpha)
    print("len(alpha) = {}".format(len(alpha)))
    
    y_predict = kperceptron_test(X_test,kernel,alpha,sv,sv_y)
    
    
    # correct
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

if __name__ == "__main__":
    main()
