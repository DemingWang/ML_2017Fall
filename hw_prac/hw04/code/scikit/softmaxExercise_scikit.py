#!python
# -*- coding: utf-8 -*-#
"""
Logistic Regression (multinomial) with scikit learn.

@author: Bhishan Poudel

@date:  Oct 16, 2017

@email: bhishanpdl@gmail.com

Ref: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

In this program we use LogisticRegression from sklearn.linear_model to fit the
data. Note that while using logistic regression we need the data matrix 
should have the shape n_samples * n_cases .
For example we use the training data from mnist (55k for train, 5k for validation)
then X_train.shape = (55000, 784).

Each row has 28 * 28 = 784 features, where 28,28 is the pixel size of given image.
"""
# Imports
import argparse
import sys
import numpy as np
from sklearn import linear_model



def softmax_sklearn():
    
    # Argument parsing
    parser = argparse.ArgumentParser('Softmax Exercise.')
    parser.add_argument('-i', '--input_data_dir',
                        type=str,
                        default='../../data/mnist/',
                        help='Directory to put the input MNIST data.')
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Used for gradient checking.')

    FLAGS, unparsed = parser.parse_known_args()

    # load data
    # NOTE: we need to transpose the X data
    X_train = np.load(FLAGS.input_data_dir + 'train-images.npy').T
    y_train = np.load(FLAGS.input_data_dir + 'train-labels.npy')
    print("X_train.shape = {}".format(X_train.shape)) # (55000, 784)
    print("y_train.shape = {}".format(y_train.shape)) # (55000,)

    # softmax regression using sklearn
    decay = 1e-4 # Weight decay parameter
    C = 1/ decay # C parameter of scikit logistic regression

    # regressor object
    import multiprocessing
    n_jobs =  multiprocessing.cpu_count()
    print("n_jobs = {}".format(n_jobs))  # 4
    softmax = linear_model.LogisticRegression(C=C, penalty='l2', random_state=100,
                  solver='lbfgs', max_iter=100, multi_class='multinomial',n_jobs=n_jobs, verbose=1)

    softmax.fit(X_train, y_train)

    # After fitting the regressor, now test the model
    # NOTE: we need to transpose the X data
    X_test = np.load(FLAGS.input_data_dir + 'test-images.npy').T
    y_test = np.load(FLAGS.input_data_dir + 'test-labels.npy')
    print("X_test.shape = {}".format(X_test.shape)) # (10000, 784)
    print("y_test.shape = {}".format(y_test.shape)) # (10000,)

    # predict
    y_pred = softmax.predict(X_test)
    acc = np.mean(y_test == y_pred)
    print('Accuracy: %0.3f%%.' % (acc * 100)) # 92.670%.

def main():
    """Run main function."""
    softmax_sklearn()


if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    #  Run the main program
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
