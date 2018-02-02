#!python
# -*- coding: utf-8 -*-#
"""
Grid stamfordresearch

@author: Bhishan Poudel

@date: Nov 18, 2017
https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
"""
# Imports
import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

# Grid Search for Algorithm Tuning
def randomized_search():
    # load the diabetes datasets
    dataset = datasets.load_diabetes()
    
    # prepare a uniform distribution to sample for the alpha parameter
    param_grid = {'alpha': sp_rand()}
    
    # create and fit a ridge regression model, testing random alpha values
    model = Ridge()
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
    rsearch.fit(dataset.data, dataset.target)
    
    # print(rsearch)
    # summarize the results of the random parameter search
    # print(rsearch.best_score_)
    print(rsearch.best_estimator_.alpha)

def main():
    """Run main function."""
    randomized_search()

if __name__ == "__main__":
    main()
