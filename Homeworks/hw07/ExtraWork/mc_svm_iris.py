#!python
# -*- coding: utf-8 -*-#
"""
Multiclass SVM classifier

@author: Bhishan Poudel

@date:  Nov 29, 2017

"""
# Imports
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import sys
sys.stdout = open('svm_iris_outputs.txt','w')

def mc_svm():     
    # get data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
     
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
     
    # fit the classifier
    model = SVC(kernel = 'linear', C = 5,random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
     
    # model accuracy for X_test  
    acc = model.score(X_test, y_test)
     
    # creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("cm = \n{}".format(cm))
    print("accuracy = {}".format(acc))
    
def main():
    """Run main function."""
    mc_svm()

if __name__ == "__main__":
    main()
