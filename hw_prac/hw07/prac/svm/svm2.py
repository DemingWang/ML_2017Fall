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
    x1 = [1, 7, 3, 1.5, 7.5, 3.8,  4, 1.5, 8]
    x2 = [2, 8, 4, 1.8, 8.6, 4.2,  5, 2.2, 8.8]
    y  = [0, 1, 2, 0,   1,   2,    2, 0,   1]

    X = np.c_[x1, x2]
    
    plt.scatter(x1,x2,c=y)
    # plt.show()
    plt.close()


    # clf = svm.SVC(kernel='linear', C = 1.0)
    clf = svm.SVC(kernel='poly', C = 1.0)
    clf.fit(X,y)
    
    X_test = np.array([[0.58, 0.76],
                       [4,   4.8],
                       [0.5, 0.5],
                       [7.7, 7.7]])

    y_pred  = clf.predict(X_test)
    print("y_pred = {}".format(y_pred))

def main():
    """Run main function."""
    svm_eg1()

if __name__ == "__main__":
    main()
