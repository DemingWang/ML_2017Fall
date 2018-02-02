#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 22, 2017
# Last update :
###########################################################################
"""
:Topic: Mutiple correlation coefficient

:Ref: https://stackoverflow.com/questions/17727954/multiple-linear-regression-python

correlation coefficient is given by:

R = np.sqrt( ((var - X.dot(a))**2).sum() )
"""
# Imports


import numpy as np

def corr():
    x1 = np.array([1,2,3,4,5,6])
    x2 = np.array([1,1.5,2,2.5,3.5,6])
    x3 = np.array([6,5,4,3,2,1])
    y = np.random.random(6)

    nvar = 3
    one = np.ones(x1.shape)
    A = np.vstack((x1,one,x2,one,x3,one)).T.reshape(nvar,x1.shape[0],2)

    for i,Ai in enumerate(A):
        a = np.linalg.lstsq(Ai,y)[0]
        R = np.sqrt( ((y - Ai.dot(a))**2).sum() )
        print (R)

def main():
    """Run main function."""
    corr()

if __name__ == "__main__":
    main()
