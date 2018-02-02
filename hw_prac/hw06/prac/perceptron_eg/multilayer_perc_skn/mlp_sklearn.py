#!python
# -*- coding: utf-8 -*-#
"""
Multi-class Perceptron sklearn

@author: Bhishan Poudel

@date: Nov 14, 2017
https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/

"""
# Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

import pandas as pd

def mlp_sklearn():

    names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"]

    wine = pd.read_csv('wine_data.txt', names=names)
    # print("wine.head() = {}".format(wine.head()))

    a = wine.describe().transpose()
    # print(a)

    # print("wine.shape = {}".format(wine.shape)) # (178, 14)

    X = wine.drop('Cultivator',axis=1)
    y = wine['Cultivator']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()

    # Fit only to the training data
    scaler.fit(X_train)

    StandardScaler(copy=True, with_mean=True, with_std=True)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # test
    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=500,random_state=100)
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)


    # print(confusion_matrix(y_test,predictions))
    # print(classification_report(y_test,predictions))

    # coefs and weights
    print("len(mlp.coefs_) = {}".format(len(mlp.coefs_)))
    print("len(mlp.coefs_[0]) = {}".format(len(mlp.coefs_[0])))
    print("len(mlp.intercepts_[0]) = {}".format(len(mlp.intercepts_[0])))

def main():
    """Run main function."""
    mlp_sklearn()

if __name__ == "__main__":
    main()
