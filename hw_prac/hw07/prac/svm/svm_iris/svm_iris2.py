#!python
# -*- coding: utf-8 -*-#
"""
smv iris datasets

@author: Bhishan Poudel

@date: Nov 19, 2017
https://sadanand-singh.github.io/posts/svmpython/

The iris dataset is a simple dataset of contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other two; the latter are NOT linearly separable from each other. Each instance has 4 features:

sepal length
sepal width
petal length
petal width
A typical problem to solve is to predict the class of the iris plant based on these 4 features. For brevity and visualization, in this example we will be using only the first two features.
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

def svm_iris1():
    iris = datasets.load_iris()
    X = iris.data[:, :2] # we only take the first two features.
    y = iris.target

    # shuffle the dataset
    X, y = shuffle(X, y, random_state=0)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Set the parameters by cross-validation
    parameters = [{'kernel': ['rbf'],
                   'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                    'C': [1, 10, 100, 1000]},
                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    print("# Tuning hyper-parameters")
    print()

    clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    #
    # plot details
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    
def main():
    """Run main function."""
    svm_iris1()

if __name__ == "__main__":
    main()
