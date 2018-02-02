#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 20, 2017
# Last update :
###########################################################################
"""
:Topic: Numpy stacks

:Runtime:

"""
# Imports
import numpy as np

# NOTE: The use of np._c and np.r_ is unpythonic
#       and syntatic sugar for matlab users.
# Use of np.concatenate is preferred.
def stacks1():
    m = np.zeros((1, 4))
    v = np.ones((1, 1))

    # column stack
    c = np.c_[m, v]
    c = np.hstack([m, v])
    np.column_stack([m, v])
    print('c = ', c)

    # row stack
    c = np.r_[1:5, 2]
    print('c = ', c)

def stacks2():
    m=np.zeros((10,4))
    v=np.ones((10,1))

    c = np.c_[v,m]
    print('c = ', c)

def np_append():
    x = np.array([[10,20,30], [40,50,60]])
    y = np.array([[100], [200]])
    print(np.append(x, y, axis=1))

def np_concatenate():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    c = np.concatenate((a, b), axis=0)
    d = np.concatenate((a, b.T), axis=1)
    print('c = \n', c)
    print('d = \n', d)

def np_stack():
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    c = np.stack((a, b))
    print('c = ', c)

    d = np.stack((a, b), axis=-1)
    print("d = ", d)

def main():
    """Run main function."""
    # stacks2()
    # np_append()
    # np_concatenate()
    np_stack()

if __name__ == "__main__":
    main()
