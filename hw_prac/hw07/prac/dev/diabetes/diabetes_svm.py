#!python
# -*- coding: utf-8 -*-#
"""
Diabetes data

@author: Bhishan Poudel

@date: Nov 20, 2017

Ref:
https://www.programcreek.com/python/example/85913/sklearn.datasets.load_diabetes

Diabetes:
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
from sklearn import datasets,linear_model
from sklearn import svm
import nose

def test_svr():
    """
    Test Support Vector Regression
    """

    diabetes = datasets.load_diabetes()
    for clf in (svm.NuSVR(kernel='linear', nu=.4, C=1.0),
                svm.NuSVR(kernel='linear', nu=.4, C=10.),
                svm.SVR(kernel='linear', C=10.),
                svm.LinearSVR(C=10.),
                svm.LinearSVR(C=10.),
                ):
        clf.fit(diabetes.data, diabetes.target)
        nose.tools.assert_greater(clf.score(diabetes.data, diabetes.target), 0.02)

    # non-regression test; previously, BaseLibSVM would check that
    # len(np.unique(y)) < 2, which must only be done for SVC
    svm.SVR().fit(diabetes.data, np.ones(len(diabetes.data)))
    svm.LinearSVR().fit(diabetes.data, np.ones(len(diabetes.data)))
    
def main():
    """Run main function."""
    test_svr()

if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    # Run the main program
    main()

    # Print the time taken
    program_end_time = time.time()
    end_ctime        = time.ctime()
    seconds          = program_end_time - program_begin_time
    m, s             = divmod(seconds, 60)
    h, m             = divmod(m, 60)
    d, h             = divmod(h, 24)
    print("\nBegin time: ", begin_ctime)
    print("End   time: ", end_ctime, "\n")
    print("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
