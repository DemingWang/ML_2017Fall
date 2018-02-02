#!python
# -*- coding: utf-8 -*-#
"""
multiclass svm sklearn

@author: Bhishan Poudel

@date: Nov 19, 2017
http://scikit-learn.org/stable/modules/svm.html
"""
# Imports
from sklearn import svm
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def mcsvm1():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    
def main():
    """Run main function."""
    mcsvm1()

if __name__ == "__main__":
    main()
