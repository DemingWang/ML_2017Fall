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

from perceptron import perceptron

from kperceptron import linear_kernel
from kperceptron import polynomial_kernel
from kperceptron import gaussian_kernel
from kperceptron import square_kernel

from kperceptron import kperceptron_train
from kperceptron import kperceptron_test
from kperceptron import read_data
from kperceptron import Gram_matrix

import sys

def multiclass_perceptron(data_pth_train,epoch,data_pth_devel_test,nclass):
    """
    Training data ==>  data_pth_train+ str(i) + '.txt' 
    """
    
    # read devel data
    fdata = data_pth_devel_test+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    
    # add bias column to devel data
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    
    # initialize big weights matrix
    weights = np.zeros([X.shape[1], nclass])
        
    # fit the model
    for i in range(nclass):
        fftrain           = data_pth_train+ str(i) + '.txt'
        X_train, t_train  = read_data(fftrain)
        
        # get weight
        w = perceptron(X_train, t_train,epoch)        
        
        # normalize w
        w = w / np.linalg.norm(w)
        weights[:,i] = w
    
    ##get the prediction
    hypothesis    =  X.dot(weights)
    t_pred = np.argmax(hypothesis,axis=1)
    
    # accuracy
    correct = np.sum(t_pred==t_actu)
    accuracy = correct / len(t_actu) * 100
    
    cm = confusion_matrix(t_actu,t_pred)
    return accuracy, cm

def multiclass_kperceptron(data_pth_train,epochs,data_pth_devel_test,kernel, nclass,kparam,K):
    """
    get alpha from train, use alpha on devel and get d.
    get alpha from train, use alpha on devel and get sigma.
   
    use d and sigma on train data.
    predict value for test.
    
    K is Gram matrix from train data
    kernel is used to get accuracy on test/devel data.
     """
    # read devel/test to find accuracy
    fdev_test = data_pth_devel_test+'.txt'
    X_dt, X_dt1, y_true = read_data(fdev_test)
    
    # initialize big hypothesis matrix (n_samples,nclass)
    hypothesis = np.zeros([X_dt1.shape[0], nclass])
    
    # initilaize support vector
    sv = []
  
    # fit the model
    for i in range(nclass):
        ftrain   = data_pth_train + str(i) + '.txt'
        X_train, X1_train, y_train  = read_data(ftrain)
        
        # get alpha from train (also use gram matrix for train)
        alpha, sv, sv_y = kperceptron_train(X1_train,y_train,epochs,K)
        
        # use alpha on devel/test data to get accuracy
        _, h = kperceptron_test(X_dt1,kernel,kparam,alpha,sv,sv_y)

        hypothesis[:,i] = h
    
    # get the prediction
    y_pred = np.argmax(hypothesis,axis=1)
    
    # accuracy
    correct = np.sum(y_true==t_pred)
    accuracy = correct / len(y_true) * 100
    
    cm = confusion_matrix(y_true,y_pred)
    return accuracy, len(sv), cm

def tune_T(data_pth,tune_epochs,data_pth_devel_test,nclass):
    sys.stdout = open('outputs/tune_epochs.txt','w')
    accuracies = []
    print("Tuning T")
    for epoch in tune_epochs:
        accuracy,_ = multiclass_perceptron(data_pth,epoch,data_pth_devel_test,nclass)
        accuracies.append(accuracy)
        print("epoch {}: accuracy = {:.2f} %".format(epoch, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_epoch = tune_epochs[best_idx]
    print("tuned epoch = {}".format(tuned_epoch))
    print("\n",'='*40)
    
    return tuned_epoch

def tune_d(data_pth,epochs,data_pth_devel_test,nclass,tune_kparams):
    sys.stdout = open('outputs/tune_d.txt','w')
    print("Tuning degree of polynomial **d** ")
    accuracies = []
    kernel = polynomial_kernel
    
    # gram matrix for train will give alpha, we use alpha on devel data.
    fdata = data_pth+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    
    # append ones
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    K = Gram_matrix(X,kernel,1)
    
    for kparam in tune_kparams:
        # Update K
        K = K**kparam
        
        accuracy,_,_ = multiclass_kperceptron(data_pth,epochs,data_pth_devel_test,kernel,nclass,kparam,K)
        accuracies.append(accuracy)
        print("degree {}: accuracy = {:.2f} %".format(kparam, accuracy))
    
    best_idx = np.argmax(accuracies)
    tuned_d = tune_kparams[best_idx]
    print("tuned d = {}".format(tuned_d))
    print("\n",'='*40)
    return tuned_d

def tune_sigma(data_pth,epochs,data_pth_devel_test,nclass,tune_kparams):
    sys.stdout = open('outputs/tune_sigma.txt','w')
    print("Tuning sigma ")
    accuracies = []
    kernel = gaussian_kernel
    
    # Unibased X for Gram matrix
    fdata = data_pth+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X = np.array(X).T.astype(np.float64)
    
    # pseduo gram matrix
    K = Gram_matrix(X,square_kernel,1)    
    
    for kparam in tune_kparams:
        # K = Gram_matrix(X,gaussian_kernel,kparam)   # slower method 
        K = np.exp(-0.5* K/kparam**2) # faster method
        
        accuracy,_,_ = multiclass_kperceptron(data_pth,epochs,data_pth_devel_test,kernel,nclass,kparam, K)
        accuracies.append(accuracy)
        print("sigma {}: accuracy = {:.2f} %".format(kparam, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_sigma = tune_kparams[best_idx]
    print("tuned sigma = {}".format(tuned_sigma))
    print("\n",'='*40)
    return tuned_sigma  

def test_mperceptron(data_pth_train,data_pth_devel_test,nclass,tuned_epoch):
    sys.stdout = open('outputs/mcp.txt','w')    
    acc, cm = multiclass_perceptron(data_pth_train,tuned_epoch,data_pth_devel_test,nclass)
    print("Multiclass Percpetron without Kernel")
    print("Accuracy = {:.2f} %".format(acc))
    # print("Number of support vectors = {}".format(len_sv)) # no sv for no kernel
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def test_mkperceptron_linear(data_train,data_pth_test,nclass,tuned_epoch):
    sys.stdout = open('outputs/mcp_klin.txt','w')
    data_train = data_train[0:-4]  
    kernel = linear_kernel
    
    # Get X for Gram matrix
    fdata = data_train+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    
    # append ones
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X].astype(np.float64)
    
    # Gram matrix
    K =  Gram_matrix(X,linear_kernel,1)
          
    acc, len_sv, cm = multiclass_kperceptron(data_train,tuned_epoch,data_pth_test,kernel,nclass,1,K)
    print("Multiclass Percpetron Linear Kernel")
    print("Accuracy = {:.2f} %".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def test_mkperceptron_poly(data_train,data_pth_test,nclass,tuned_epoch,tuned_d):
    sys.stdout = open('outputs/mcp_kpoly.txt','w')
    data_train = data_train[0:-4]
    kernel = polynomial_kernel
    kparam = tuned_d
    
    # Get unbiased X for Gram matrix
    fdata = data_train+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X = np.array(X).T
    
    # Gram matrix
    K =  Gram_matrix(X,polynomial_kernel,kparam)
        
    acc, len_sv, cm = multiclass_kperceptron(data_train,tuned_epoch,data_pth_test,kernel,nclass,kparam,K)
    print("Multiclass Percpetron Polynomial Kernel")
    print("Accuracy = {:.2f} %".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def test_mkperceptron_gau(data_train,data_pth_test,tuned_epoch,tuned_sigma,nclass):
    sys.stdout = open('outputs/mcp_kgau.txt','w')
    data_train = data_train[0:-4]
    kernel = gaussian_kernel
    epochs = tuned_epoch
    kparam = tuned_sigma
    
    
    # Get unbiased X for Gram matrix
    ftrain = data_train+'.txt'
    X_train, X1_train, y_train = read_data(ftrain)
    
    # Gram matrix
    K =  Gram_matrix(X1_train,gaussian_kernel,kparam)    
        
    acc, len_sv, cm = multiclass_kperceptron(data_train,epochs,data_pth_test,kernel,nclass,kparam,K)
    print("Multiclass Percpetron Gaussian Kernel")
    print("Accuracy = {:.2f} %".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def main():
    """Run main function."""
    # data path
    data_train = '../data/optdigits/train/train.txt'
    data_devel = '../data/optdigits/train/devel.txt'
    data_test = '../data/optdigits/train/test.txt'
    
    data_pth_train = '../data/optdigits/train/train'
    data_pth_devel = '../data/optdigits/devel/devel'
    data_pth_test = '../data/optdigits/test/test'
    
    # create output folder
    if not os.path.isdir('outputs'):
        os.makedirs('outputs')
    
    # tuning params
    nclass=10
    tune_epochs = list(range(1,21))
    tune_kparams = [2,3,4,5,6]
    tune_kparams2 = [0.1,0.5,2,5,10]
    tuned_epoch = 4 # 4 for optdigits, 19 for iris, takes 1/2 min to tune
    tuned_d = 6 # 6 for optdigits takes 10 min
    tuned_sigma = 10
    
    # tuning (train from train, test on devel)
    # tuned_epoch = tune_T(data_pth_train,tune_epochs,data_pth_devel,nclass)
    # tuned_d = tune_d(data_pth_train,tuned_epoch,data_pth_devel,nclass,tune_kparams)
    # tuned_sigma = tune_sigma(data_pth_train,tuned_epoch,data_pth_devel,nclass,tune_kparams2)
    
    # testing (train from train, test on test)
    tuned_epoch = 4
    tuned_d = 5
    tuned_sigma = 0.5
    # test_mperceptron(data_pth_train,data_pth_test,nclass,tuned_epoch)
    # test_mkperceptron_linear(data_train,data_pth_test,nclass,tuned_epoch)
    # test_mkperceptron_poly(data_train,data_pth_test,nclass,tuned_epoch,tuned_d)
    test_mkperceptron_gau(data_train,data_pth_test,tuned_epoch,tuned_sigma,nclass)
   

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
