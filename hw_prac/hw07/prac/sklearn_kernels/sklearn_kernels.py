#!python
# -*- coding: utf-8 -*-#
"""
Kernel Trick

@author: Bhishan Poudel

@date: Nov 18, 2017

http://scikit-learn.org/stable/modules/metrics.html#metrics
"""
# Imports
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel


X = np.array([[0, 1], [1, 0]])
y = np.array([[0, 1], [1,2]])
K = linear_kernel(X,y)

print("X.shape = {}".format(X.shape))
print("y.shape = {}".format(y.shape))
print("K = \n{}".format(K))
