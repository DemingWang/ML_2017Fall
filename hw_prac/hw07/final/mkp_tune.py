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

from sklearn.metrics import confusion_matrix

from kperceptron import linear_kernel
from kperceptron import polynomial_kernel
from kperceptron import gaussian_kernel

from kperceptron import kperceptron_train
from kperceptron import kperceptron_test
from kperceptron import read_data

import sys


def multiclass_kperceptron(dtrain,d_dev_test,epoch,nclass,kernel,kparam):
    """
    get alpha from train, use alpha on devel and get d.
    get alpha from train, use alpha on devel and get sigma.
   
    use d and sigma on train data.
    predict value for test.
     """
    # read devel/test to find accuracy
    f_dev_test   = d_dev_test+'.txt'
    X_dt, y_true = read_data(f_dev_test)
    
    # initialize big hypothesis matrix (n_samples,nclass)
    hypothesis = np.zeros([X_dt.shape[0], nclass]).astype(np.float64)
  
    # fit the model
    for i in range(nclass):
        ftrain   = dtrain + str(i) + '.txt'
        X_train, y_train  = read_data(ftrain)
        
        # get alpha from train
        alpha, sv, sv_y = kperceptron_train(X_train,y_train,epoch,kernel,kparam)
        
        # use alpha on devel/test data to get accuracy
        h, _ = kperceptron_test(X_dt,kernel,alpha,sv,sv_y,kparam)
        
        # delete loop variables after use (sv is needed)
        del alpha; del sv_y
        
        # clear all but last sv
        if i < nclass-1:
            del sv

        hypothesis[:,i] = h
    
    # get the prediction
    y_pred = np.argmax(hypothesis,axis=1)
    
    # debug
    np.savetxt('tmp.csv',hypothesis,fmt='%.4g',delimiter=',')
    np.savetxt('tmp2.csv',y_pred,fmt='%.4g',delimiter=',')
    
    # accuracy
    correct = np.sum(y_true==y_pred)
    accuracy = correct / len(y_true) * 100
    
    cm = confusion_matrix(y_true,y_pred)
    return accuracy, len(sv), cm

def tune_kernel(dtrain,ddevel,tuned_epoch,nclass,kernel,tune_kparams,kname):
    sys.stdout = open('outputs/tune_{}.txt'.format(kname),'w')
    
    accuracies = []
    print("Tuning Kernel parameter {}".format(kname))
    for kparam in tune_kparams:
        accuracy,_,_ = multiclass_kperceptron(dtrain,ddevel,tuned_epoch,nclass,kernel,kparam)
        accuracies.append(accuracy)
        print("{} {}: accuracy = {:.2f}".format(kname, kparam, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_val = tune_kparams[best_idx]
    print("tuned {} = {}".format(kname, tuned_val))
    
    return tuned_val
 

def test_mkperceptron(dtrain,dtest,epoch,nclass,kernel,kparam, kname):
    sys.stdout = open('outputs/mcp_{}.txt'.format(kname),'w')

    acc, len_sv, cm = multiclass_kperceptron(dtrain,dtest,epoch,nclass,kernel,kparam)
    print("Multiclass Percpetron {} Kernel".format(kname))
    print("Accuracy = {:.2f} %".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def main():
    """Run main function."""
    # data path    
    dtrain = '../data/optdigits/train/train'
    ddevel = '../data/optdigits/devel/devel'
    dtest  = '../data/optdigits/test/test'
    
    # create output folder
    if not os.path.isdir('outputs'):
        os.makedirs('outputs')
    
    # testing (train from train, test on test)
    nclass=10
    tuned_epoch = 19
    tuned_d = 5
    tuned_sigma = 5.0
    
    # tuning 
    tune_kparams = [2,3,4,5,6]
    tune_kparams2 = [0.1, 0.5, 2, 5, 10]
    
    
    # tune_kernel(dtrain,ddevel,tuned_epoch,nclass,polynomial_kernel,
                  # tune_kparams2,'sigma')
    # test_mkperceptron(dtrain,dtest,tuned_epoch,nclass,
    #                       polynomial_kernel,tuned_d,'polynomial')
    
    
    tuned_sigma = tune_kernel(dtrain,ddevel,tuned_epoch,nclass,
                              gaussian_kernel,tune_kparams2,'sigma')
    
    test_mkperceptron(dtrain,dtest,tuned_epoch,nclass,
                          gaussian_kernel,tuned_sigma,'gaussian')
    


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
