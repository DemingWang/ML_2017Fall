#!python
# -*- coding: utf-8 -*-#
"""
svm

@author: Bhishan Poudel

@date: Nov 19, 2017

"""
# Imports
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification


def eg1():
    X, y = make_classification(n_features=4, random_state=0)
    clf = LinearSVC(random_state=0)
    clf.fit(X, y)

    print((clf.coef_))

    print((clf.intercept_))

    print((clf.predict([[0, 0, 0, 0]])))
    
def main():
    """Run main function."""
    eg1()

if __name__ == "__main__":
    main()    
