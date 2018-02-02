#!python
# -*- coding: utf-8 -*-#
"""
K class perceptron

@author: Bhishan Poudel

@date: Oct 29, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import trange
import subprocess

# Generate three random clusters of 2D data
def gen_data():
    N_c = 200
    K = 3
    N = N_c * K
    A = 0.6*np.random.randn(N_c, 2)+[1, 1]
    B = 0.6*np.random.randn(N_c, 2)+[3, 3]
    C = 0.6*np.random.randn(N_c, 2)+[3, 0]
    X = np.hstack((np.ones(3*N_c).reshape(3*N_c, 1), np.vstack((A, B, C))))
    Y = np.vstack(((np.zeros(N_c)).reshape(N_c, 1), np.ones(N_c).reshape(N_c, 1), 2*np.ones(N_c).reshape(N_c, 1)))
  
    return X,Y,N_c,N

# Run gradient descent
def grad_desc(X,Y,N,eta,max_iter):
    w = np.zeros((3, 3)) # initiliaze w
    w_lst = []
    for t in range(0, max_iter):
        grad_t = np.zeros((3, 3)) # temporary gradient
        for i in range(0, N):
            x_i = X[i, :]
            y_i = Y[i]
            exp_vals = np.exp(w.dot(x_i))
            lik = exp_vals[int(y_i)]/np.sum(exp_vals)
            grad_t[int(y_i), :] += x_i*(1-lik)

        w = w + 1/float(N)*eta*grad_t
        w_lst.append(w)
            
    return w_lst


# Begin plotting here
def myplot(X,Y,w,N_c,max_iter):
    # Define our class colors
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])

    # Generate the mesh
    x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    y_min, y_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    h = 0.02 # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_mesh = np.c_[np.ones((xx.size, 1)), xx.ravel(), yy.ravel()]
    Z = np.zeros((xx.size, 1))

    # Compute the likelihood of each cell in the mesh
    for i in range(0, xx.size):
        lik = w.dot(X_mesh[i, :])
        Z[i] = np.argmax(lik)

    # Plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.plot(X[0:N_c-1, 1], X[0:N_c-1, 2], 'ro', X[N_c:2*N_c-1, 1], X[N_c:2*N_c-1,
        2], 'bo', X[2*N_c:, 1], X[2*N_c:, 2], 'go')
    plt.axis([np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5, np.min(X[:, 2])-0.5, np.max(X[:, 2])+0.5])

    plt.title('iteration = {}'.format(max_iter))
    plt.savefig('img/iter_{:03d}.png'.format(max_iter))
    plt.close()

def create_gif():
    cmd = "convert -loop 0 -delay 100 img/*.png img.gif;open img.gif"
    subprocess.call(cmd, shell=True)

def main():
    """Run main function."""
    X,Y,N_c,N = gen_data()
    eta = 1
    max_iter = 100
    step = 2
    
    w_lst = grad_desc(X,Y,N,eta,max_iter)
    
    for max_iter in trange(0,100,step): 
        myplot(X,Y,w_lst[max_iter],N_c,max_iter) 
    
    # now create gif
    create_gif()
    

if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    # Run the main program
    main()

    # Print the time taken
    program_end_time = time.time()
    end_ctime        = time.ctime()
    seconds          = program_end_time - program_begin_time
    m, s             = divmod(seconds, 60)
    h, m             = divmod(m, 60)
    d, h             = divmod(h, 24)
    print(("\nBegin time: ", begin_ctime))
    print(("End   time: ", end_ctime, "\n"))
    print(("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s)))
