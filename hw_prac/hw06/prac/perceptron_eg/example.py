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

def main():
    """Run main function."""

    data = np.loadtxt('data.txt')
    # print("data = \n{}".format(data))
    
    X0 = []
    X1 = []
    
    for i in range(len(data)):
        if data[i][-1] == -1:
            X0.append(data[i][0:-1])
        if data[i][-1] == 1:
            X1.append(data[i][0:-1])

    print(X1)
    # Example of -1
    #a = [1,2,3,-1]
    # print("a[0] = {}".format(a[0]))
    # print("a[3] = {}".format(a[3]))
    # print("a[-1] = {}".format(a[-1]))

if __name__ == "__main__":
    main()


"""
#data.txt
-2 4  -1
4  1  -1
1  6   1
-2 1  -1
2  4   1
6  2   1

"""
