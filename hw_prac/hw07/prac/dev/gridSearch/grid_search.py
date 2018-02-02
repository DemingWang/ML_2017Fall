#!python
# -*- coding: utf-8 -*-#
"""
Grid stamfordresearch

@author: Bhishan Poudel

@date: Nov 18, 2017
https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/

diabetes data
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/descr/diabetes.rst

Ten baseline variables, age, sex, body mass index, average blood pressure, 
and six blood serum measurements were obtained for each of n = 442 diabetes 
patients, as well as the response of interest, a quantitative measure of 
disease progression one year after baseline.

Target:	
Column 11 is a quantitative measure of disease progression one year after baseline

Note: Each of these 10 feature variables have been mean centered and scaled by 
the standard deviation times n_samples 
(i.e. the sum of squares of each column totals 1).
"""
# Imports
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import perceptron



# Grid Search for Algorithm Tuning
def grid_search():
    # load the diabetes datasets
    dataset = datasets.load_diabetes()
    X = dataset.data
    y = dataset.target
    
    print("X.shape = {}".format(X.shape))
    print("y.shape = {}".format(y.shape))
    print("X[0] = {}".format(X[0:10]))
    print("y[0:10] = {}".format(y[0:10]))
    
    # prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    
    # create and fit a ridge regression model, testing each alpha
    model = Ridge()
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    grid.fit(X, y)
    
    # print(grid)
    # summarize the results of the grid search
    # print(grid.best_score_)
    best_alpha = grid.best_estimator_.alpha
    print("best_alpha = {}".format(best_alpha))
    
def grid_search2():
    # load the diabetes datasets
    dataset = datasets.load_diabetes()
    
    # prepare a range of alpha values to test
    max_iters = np.arange(1,20)
    
    # create and fit a ridge regression model, testing each alpha
    model = perceptron.Perceptron()
    grid = GridSearchCV(estimator=model, param_grid=dict(max_iter=max_iters))
    grid.fit(dataset.data, dataset.target)
    
    # print(grid)
    # summarize the results of the grid search
    # print(grid.best_score_)
    print(grid.best_estimator_.max_iter)

def main():
    """Run main function."""
    grid_search()

if __name__ == "__main__":
    main()
