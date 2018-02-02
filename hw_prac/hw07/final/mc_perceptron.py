#!python
# -*- coding: utf-8 -*-#
"""
Multiclass perceptron

@author: Bhishan Poudel

@date: Nov 18, 2017

"""
# Imports
import numpy as np
import os
import sys

from sklearn.metrics import confusion_matrix

from perceptron import perceptron

def read_data(fdata):
    *X,y = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X = np.array(X).T
    X1 = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]

    return X, X1, y

def multiclass_perceptron(dtrain,ddevtest,epoch):
    """
    Training data ==>  dtrain+ str(i) + '.txt'
    """

    # read devel data
    _, X1dt, ydt = read_data(ddevtest+'.txt')

    # initialize big weights matrix
    nclass = 10
    weights = np.zeros([X1dt.shape[1], nclass])

    # fit the model
    for i in range(nclass):
        ftrain           = dtrain+ str(i) + '.txt'
        _, X1_train, t1_train  = read_data(ftrain)

        # get weight
        w = perceptron(X1_train, t1_train,epoch)

        # normalize w
        w = w / np.linalg.norm(w)
        weights[:,i] = w

    # get the prediction
    hypothesis    =  X1dt.dot(weights)
    y_pred = np.argmax(hypothesis,axis=1)

    # accuracy
    correct = np.sum(y_pred==ydt)
    accuracy = correct / len(ydt) * 100

    cm = confusion_matrix(ydt,y_pred)
    return accuracy, cm

def tune_T(dtrain,ddevtest,tune_epochs):
    sys.stdout = open('outputs/tune_T.txt','w')
    accuracies = []
    for epoch in tune_epochs:
        accuracy,_ = multiclass_perceptron(dtrain,ddevtest,epoch)
        accuracies.append(accuracy)
        print("epoch {}: accuracy = {:.2f}".format(epoch, accuracy))

    best_idx = np.argmax(accuracies)
    tuned_epoch = tune_epochs[best_idx]
    print("tuned epoch = {}".format(tuned_epoch))

    return tuned_epoch

def test_mperceptron(dtrain,ddevtest,tuned_epoch):
    sys.stdout = open('outputs/test_mperceptron.txt','w')

    acc, cm = multiclass_perceptron(dtrain,ddevtest,tuned_epoch)

    print("Accuracy = {}".format(acc))
    # print("Number of support vectors = {}".format(len_sv)) # no sv
    print("\n")
    print(cm)

def main():
    """Run main function."""
    # data path
    dtrain = '../data/optdigits/train/train'
    ddevel = '../data/optdigits/devel/devel'
    dtest = '../data/optdigits/test/test'

    # create output folder
    if not os.path.isdir('outputs'):
        os.makedirs('outputs')

    # tuning
    tune_epochs = list(range(1,21))
    tuned_epoch = tune_T(dtrain,ddevel,tune_epochs)

    # tuned parametr
    tuned_epoch = 4

    # testing
    test_mperceptron(dtrain,dtest,tuned_epoch)


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
