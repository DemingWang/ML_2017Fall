#!python
# -*- coding: utf-8 -*-#
"""
SVM example

@author: Bhishan Poudel

@date: Nov 20, 2017
https://stackoverflow.com/questions/45554008/error-in-python-script-expected-2d-array-got-1d-array-instead
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
from sklearn import svm

def svm_eg1():
    x = [1, 5, 1.5, 8, 1, 9]
    y = [2, 8, 1.8, 8, 0.6, 11]

    plt.scatter(x,y)
    plt.show()
    plt.close()

    X = np.array([[1,2],
                 [5,8],
                 [1.5,1.8],
                 [8,8],
                 [1,0.6],
                 [9,11]])

    y = [0,1,0,1,0,1]

    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X,y)

    test = np.array([0.58, 0.76])
    print(test)       # Produces: [ 0.58  0.76]
    print(test.shape) # Produces: (2,) meaning 2 rows, 1 col

    test = test.reshape(1, -1)
    print(test)       # Produces: [[ 0.58  0.76]]
    print(test.shape) # Produces (1, 2) meaning 1 row, 2 cols

    print((clf.predict(test))) # Produces [0], as expected

def main():
    """Run main function."""
    svm_eg1()

if __name__ == "__main__":
    main()
