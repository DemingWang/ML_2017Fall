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

from kperceptron import kperceptron_train
from kperceptron import kperceptron_test
from kperceptron import read_data

# log file
import sys
log_file = open('outputs.log','w')
log_file = open('outputs.log','a')
sys.stdout = log_file

def read_data(fdata):
    *X,t = np.genfromtxt(fdata,unpack=True,dtype='f')
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    
    return X,t

def multiclass_perceptron(data_pth_train,epoch,data_pth_devel_test,nclass):
    """
    Training data ==>  data_pth_train+ str(i) + '.txt' 
    """
    
    # read devel data
    fdata = data_pth_devel_test+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=' ')
    
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

def multiclass_kperceptron(data_pth_train,epochs,data_pth_devel_test,kernel, nclass,kparam):
    
    # read devel data
    fdata = data_pth_devel_test+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=' ')
    
    # append ones to devel data
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    
    # initialize big hypothesis matrix
    hypothesis = np.zeros([X.shape[0], nclass])
    
    # initilaize support vector
    sv = []
  
    # fit the model
    for i in range(nclass):
        fftrain           = data_pth_train+ str(i) + '.txt'
        X_train, t_train  = read_data(fftrain)
        
        # get alpha
        alpha, sv, sv_y = kperceptron_train(X_train,t_train,epochs,kernel,kparam)
        _, h = kperceptron_test(X,kernel,kparam,alpha,sv,sv_y)

        hypothesis[:,i] = h
    
    # get the prediction
    t_pred = np.argmax(hypothesis,axis=1)
    
    # accuracy
    correct = np.sum(t_actu==t_pred)
    accuracy = correct / len(t_actu) * 100
    
    cm = confusion_matrix(t_actu,t_pred)
    return accuracy, len(sv), cm

def tune_T(data_pth,tune_epochs,data_pth_devel_test,nclass):
    accuracies = []
    print("Tuning T")
    for epoch in tune_epochs:
        accuracy,_ = multiclass_perceptron(data_pth,epoch,data_pth_devel_test,nclass)
        accuracies.append(accuracy)
        print("epoch {}: accuracy = {:.2f}".format(epoch, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_epoch = tune_epochs[best_idx]
    print("tuned epoch = {}".format(tuned_epoch))
    print("\n",'='*40)
    
    return tuned_epoch

def tune_d(data_pth,epochs,data_pth_devel_test,nclass,tune_kparams):
    print("Tuning degree of polynomial **d** ")
    accuracies = []
    kernel = polynomial_kernel
    
    for kparam in tune_kparams:
        accuracy,_,_ = multiclass_kperceptron(data_pth,epochs,data_pth_devel_test,kernel,nclass,kparam)
        accuracies.append(accuracy)
        print("degree {}: accuracy = {:.2f}".format(kparam, accuracy))
    
    best_idx = np.argmax(accuracies)
    tuned_d = tune_kparams[best_idx]
    print("tuned d = {}".format(tuned_d))
    print("\n",'='*40)
    return tuned_d

def tune_sigma(data_pth,epochs,data_pth_devel_test,nclass,tune_kparams):
    print("Tuning sigma ")
    accuracies = []
    kernel = gaussian_kernel
    for kparam in tune_kparams:
        accuracy,_,_ = multiclass_kperceptron(data_pth,epochs,data_pth_devel_test,kernel,nclass,kparam)
        accuracies.append(accuracy)
        print("sigma {}: accuracy = {:.2f}".format(kparam, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_sigma = tune_kparams[best_idx]
    print("tuned sigma = {}".format(tuned_sigma))
    print("\n",'='*40)
    return tuned_sigma  

def test_mperceptron(data_pth_train,data_pth_devel_test,nclass,tuned_epoch):      
    acc, cm = multiclass_perceptron(data_pth_train,tuned_epoch,data_pth_devel_test,nclass)
    print("Multiclass Percpetron without Kernel")
    print("Accuracy = {:.2f}".format(acc))
    # print("Number of support vectors = {}".format(len_sv)) # no sv for no kernel
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def test_mkperceptron_linear(data_train,data_pth_test,nclass,tuned_epoch,tuned_d):
    data_train = data_train[0:-4]  
    kernel = linear_kernel        
    acc, len_sv, cm = multiclass_kperceptron(data_train,tuned_epoch,data_pth_test,kernel,nclass,tuned_d)
    print("Multiclass Percpetron Linear Kernel")
    print("Accuracy = {:.2f}".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def test_mkperceptron_poly(data_train,data_pth_test,nclass,tuned_epoch,tuned_d):
    data_train = data_train[0:-4]
    kernel = polynomial_kernel
    kparam = tuned_d
    
    acc, len_sv, cm = multiclass_kperceptron(data_train,tuned_epoch,data_pth_test,kernel,nclass,kparam)
    print("Multiclass Percpetron Polynomial Kernel")
    print("Accuracy = {:.2f}".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def test_mkperceptron_gau(data_train,data_pth_test,tuned_epoch,tuned_sigma,nclass):
    data_train = data_train[0:-4]
    kernel = gaussian_kernel
    epochs = tuned_epoch
    kparam = tuned_sigma
        
    acc, len_sv, cm = multiclass_kperceptron(data_train,epochs,data_pth_test,kernel,nclass,kparam)
    print("Multiclass Percpetron Gaussian Kernel")
    print("Accuracy = {:.2f}".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("Confusion Matrix\n")
    print(cm)
    print('\n', '='*40)

def main():
    """Run main function."""
    # data path
    data_train = '../data/iris/train/iris_train.txt'
    data_devel = '../data/iris/train/iris_devel.txt'
    data_test = '../data/iris/train/iris_test.txt'
    
    data_pth_train = '../data/iris/train/iris_train'
    data_pth_devel = '../data/iris/devel/iris_devel'
    data_pth_test = '../data/iris/test/iris_test'
    
    # actual target
    t_actu_train = np.genfromtxt(data_train, usecols=(-1),delimiter=' ')
    
    # create output folder
    if not os.path.isdir('outputs'):
        os.makedirs('outputs')
    
    # tuning
    nclass=3
    tune_epochs = list(range(1,21))
    tune_kparams = [2,3,4,5,6]
    tune_kparams2 = [0.1,0.5,2,5,10]
    tuned_epoch = 19
    tuned_d = 2
    tuned_sigma = 0.5
    
    # tuned_epoch = tune_T(data_pth_train,tune_epochs,data_pth_devel,nclass)
    # tuned_d = tune_d(data_pth_train,tuned_epoch,data_pth_devel,nclass,tune_kparams)
    # tuned_sigma = tune_sigma(data_pth_train,tuned_epoch,data_pth_devel,nclass,tune_kparams2)
    
    # testing
    tuned_epoch = 19
    tuned_d = 2
    tuned_sigma = 0.5
    test_mperceptron(data_pth_train,data_pth_test,nclass,tuned_epoch)
    # test_mkperceptron_linear(data_train,data_pth_test,nclass,tuned_epoch,tuned_d)
    # test_mkperceptron_poly(data_train,data_pth_test,nclass,tuned_epoch,tuned_d)
    # test_mkperceptron_gau(data_train,data_pth_test,tuned_epoch,tuned_sigma,nclass)
    
   

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
