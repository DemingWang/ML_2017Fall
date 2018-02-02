#!python
# -*- coding: utf-8 -*-#
"""
svm

@author: Bhishan Poudel

@date: Nov 19, 2017
https://pythonprogramming.net/support-vector-machine-svm-example-tutorial-scikit-learn-python/
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import datasets
from sklearn import svm

def svm_eg1():
    
    # load data
    digits = datasets.load_digits()
    
    # print data and target
    # print(digits.data)
    # print(digits.target)
    
    # instanciatte model
    clf = svm.SVC(gamma=0.001, C=100)

    # training data (last 10 are for testing)
    X,y = digits.data[:-10], digits.target[:-10]
    
    # train the model
    clf.fit(X,y)
    mynum = digits.data[-5][None]
    print(("mynum.shape = {}".format(mynum.shape)))
    print((clf.predict(mynum)))
    
    ##plot and see the data
    plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    plt.close()

def main():
    """Run main function."""
    svm_eg1()

if __name__ == "__main__":
    main()
