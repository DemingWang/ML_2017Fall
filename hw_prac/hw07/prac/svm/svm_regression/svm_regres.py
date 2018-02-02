#!python
# -*- coding: utf-8 -*-#
"""
smv iris datasets

@author: Bhishan Poudel

@date: Nov 19, 2017
https://sadanand-singh.github.io/posts/svmpython/

"""
# Imports
import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.svm import SVR

def svm_regr():
    X = np.sort(5 * np.random.rand(200, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(40))
    
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=3)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)

    lw = 2
    plt.figure(figsize=(12, 7))
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    plt.close()
    
def main():
    """Run main function."""
    svm_regr()

if __name__ == "__main__":
    main()
