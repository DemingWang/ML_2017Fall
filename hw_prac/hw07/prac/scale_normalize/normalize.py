#!python
# -*- coding: utf-8 -*-#
"""
Feature Scaling

@author: Bhishan Poudel

@date: Nov 17, 2017
https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.norm.html
"""
# Imports
import numpy as np
np.set_printoptions(3)
from sklearn.preprocessing import normalize

def normalize_eg():
    a = np.array([1,3,4])
    norm = np.linalg.norm(a)
    print("norm = {}".format(norm))
    
    a = a/ norm
    print('a = ', a)

    # preprocessing needs at least 2d
    a2 = normalize(a[:,np.newaxis], axis=0).ravel()
    print(a2)

def main():
    normalize_eg()

if __name__ == "__main__":
    main()
