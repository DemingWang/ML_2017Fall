#!python
# -*- coding: utf-8 -*-#
"""
Error analysis

@author: Bhishan Poudel

@date: Nov 13, 2017



"""
# Imports
import numpy as np
import itertools


def eg1():
    """  0  1
     0: [[3 0]
     1: [2 7]]
    """
    y_true = [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]

    classes = list(set(y_true)) # [0, 1]
    n = len(classes)            # 2

    # list.count gives a number
    a = list(zip(y_true,y_pred))
    # [(1, 0), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (1, 0), (1, 1), (0, 0), (1, 1), (1, 1)]
    # print('a = ', a)
    # print("a.count((0,0)) = {}".format(a.count((0,0))))
    # print("a.count((0,1)) = {}".format(a.count((0,1))))
    # print("a.count((1,0)) = {}".format(a.count((1,0))))
    # print("a.count((1,1)) = {}".format(a.count((1,1))))
    
    
    # itertools product
    # classes = [0,1]
    # classes repeat 2==> [0 1] permutation [0 1]
    #
    # (0, 0)
    # (0, 1)
    # (1, 0)
    # (1, 1)
    #
    # product(A, B) returns the same as ((x,y) for x in A for y in B)
    # for x in itertools.product(classes,repeat=2):
    #     print(x)

    # [3, 0, 2, 7]
    cm = [ list(zip(y_true,y_pred)).count(x) 
                    for x in itertools.product(classes,repeat=2)]
    
    cm = np.array(cm).reshape(n,n)
    print (cm)
    
    print("[0,1]*2 = {}".format([0,1]*2))

def main():
    """Run main function."""
    eg1()

if __name__ == "__main__":
    main()
    
