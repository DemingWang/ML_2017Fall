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

from kperceptron import linear_kernel
from kperceptron import polynomial_kernel
from kperceptron import gaussian_kernel

from kperceptron import kperceptron_train
from kperceptron import kperceptron_test
from kperceptron import read_data


def mc_perceptron(dtrain,ddevtest,epoch):
    """
    Training data ==>  data_pth_train+ str(i) + '.txt' 
    """
    
    # read devel data
    X,y_true = read_data(ddevtest + '.txt')
    
    # initialize big weights matrix
    nclass = 10
    weights = np.zeros([X.shape[1], nclass])
        
    # fit the model
    for i in range(nclass):
        fftrain           = dtrain+ str(i) + '.txt'
        X_train, t_train  = read_data(fftrain)
        
        # get weight
        w = perceptron(X_train, t_train,epoch)        
        
        # normalize w
        w = w / np.linalg.norm(w)
        weights[:,i] = w
    
    ##get the prediction
    hypothesis    =  X.dot(weights)
    y_pred = np.argmax(hypothesis,axis=1)
    
    # accuracy
    correct = np.sum(y_pred==y_true)
    accuracy = correct / len(y_true) * 100
    
    cm = confusion_matrix(y_true,y_pred)
    return accuracy, cm

def mc_kperceptron(dtrain,ddevtest,epochs,kernel, kparam):    
    # read devel data
    X,y_true = read_data(ddevtest+ '.txt')
        
    # initialize big hypothesis matrix
    nclass = 10
    hypothesis = np.zeros([X.shape[0], nclass])
    
    # initilaize support vector
    sv = []
  
    # fit the model
    for i in range(nclass):
        fftrain           = dtrain+ str(i) + '.txt'
        X_train, y_train  = read_data(fftrain)
        
        # get alpha
        alpha, sv, sv_y = kperceptron_train(X_train,y_train,epochs,kernel,kparam)
        _, h = kperceptron_test(X,kernel,kparam,alpha,sv,sv_y)

        hypothesis[:,i] = h
    
    # get the prediction
    y_pred = np.argmax(hypothesis,axis=1)
    
    # accuracy
    correct = np.sum(y_true==y_pred)
    accuracy = correct / len(y_true) * 100
    
    cm = confusion_matrix(y_true,y_pred)
    return accuracy, len(sv), cm

def tune_T(dtrain,ddev,tune_epochs,pname):
    sys.stdout = open('outputs/tune_{}.txt'.format(pname),'w')
    accuracies = []
    for epoch in tune_epochs:
        accuracy,_ = mc_perceptron(dtrain,ddev,epoch)
        accuracies.append(accuracy)
        print("epoch {}: accuracy = {:.2f}".format(epoch, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_val = tune_epochs[best_idx]
    print("tuned {} = {}".format(pname, tuned_val))
    
    return tuned_val


def tune_dsigma(dtrain,ddev,tuned_epoch,kernel,kparam):
    sys.stdout = open('outputs/tune_{}.txt'.format(kparam),'w')
    accuracies = []
    for epoch in range(tuned_epoch):
        accuracy,_,_ = mc_kperceptron(dtrain,ddev,tuned_epoch,kernel,kparam)
        accuracies.append(accuracy)
        print("epoch {}: accuracy = {:.2f}".format(epoch, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_val = tuned_epoch[best_idx]
    print("tuned {} = {}".format(kparam, tuned_val))
    
    return tuned_val

def test_mcp(dtrain,dtest,tuned_epoch,kname):
    sys.stdout = open('outputs/mc_{}.txt'.format(kname),'w')      
    acc, cm = mc_perceptron(dtrain,dtest,tuned_epoch)
    
    print("Accuracy = {}".format(acc))
#    print("Number of support vectors = {}".format(len_sv)) # no sv for mcp
    print("\n")
    print(cm)

def test_kmcp(dtrain,dtest,kernel,tuned_epoch,tuned_kparam,kname):    
    sys.stdout = open('outputs/mc_{}.txt'.format(kname),'w') 
    acc, len_sv, cm = mc_kperceptron(dtrain,dtest,tuned_epoch,kernel,tuned_kparam)
    
    print("Accuracy = {}".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("\n")
    print(cm)


def main():
    """Run main function."""
    # data path
    
    dtrain = '../data/optdigits/train/train'
    ddev = '../data/optdigits/devel/devel'
#    dtest = '../data/optdigits/test/test'
    
    # create output folder
    if not os.path.isdir('outputs'):
        os.makedirs('outputs')
    
    # tuning values
    tune_epochs = list(range(1,21))
    tune_kparams = [2, 3, 4, 5,6]
    tune_kparams2 = [0.1, 0.5, 2.0, 5.0, 10.0]
    
    # tune hyperparameters
#    tuned_epoch = tune_T(dtrain,ddev,tune_epochs,'T') # 13
    
    # tune d and sigma
    tuned_epoch = 13
    tuned_d = tune_dsigma(dtrain,ddev,tuned_epoch,polynomial_kernel,'d') # 4 
    
    # testing
    # test_mperceptron()
    # test_mkperceptron_linear()
    # test_mkperceptron_poly()
    # test_mkperceptron_gau() 
   

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
