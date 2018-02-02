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
from kperceptron import kperceptron_test,read_data


def multiclass_perceptron(data_pth_train,epoch,data_pth_devel_test):
    """
    Training data ==>  data_pth_train+ str(i) + '.txt' 
    """
    
    # read devel data
    fdata = data_pth_devel_test+'.txt'
    X,t_actu = read_data(fdata)
    
    # initialize big weights matrix
    nclass = 10
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

def multiclass_kperceptron(data_pth_train,epochs,data_pth_devel_test,kernel, kparam):    
    # read devel data
    fdata = data_pth_devel_test+'.txt'
    X,t_actu = read_data(fdata)
    
    
    # initialize big hypothesis matrix
    nclass = 10
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

def tune_T(data_pth,tune_epochs,data_pth_devel_test):
    accuracies = []
    for epoch in tune_epochs:
        accuracy,_ = multiclass_perceptron(data_pth,epoch,data_pth_devel_test)
        accuracies.append(accuracy)
        print("epoch {}: accuracy = {:.2f}".format(epoch, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_epoch = tune_epochs[best_idx]
    print("tuned epoch = {}".format(tuned_epoch))
    
    return tuned_epoch

def tune_d(data_pth,epochs,data_pth_devel_test,tune_kparams):
    accuracies = []
    kernel = polynomial_kernel
    for kparam in tune_kparams:
        accuracy,_,_ = multiclass_kperceptron(data_pth,epochs,data_pth_devel_test,kernel,kparam)
        accuracies.append(accuracy)
        print("degree {}: accuracy = {:.2f}".format(kparam, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_d = tune_kparams[best_idx]
    print("tuned d = {}".format(tuned_d))
    
    return tuned_d

def tune_sigma(data_pth,epochs,data_pth_devel_test,tune_kparams):
    accuracies = []
    kernel = gaussian_kernel
    for kparam in tune_kparams:
        accuracy,_,_ = multiclass_kperceptron(data_pth,epochs,data_pth_devel_test,kernel,kparam)
        accuracies.append(accuracy)
        print("sigma {}: accuracy = {:.2f}".format(kparam, accuracy))
        
    best_idx = np.argmax(accuracies)
    tuned_sigma = tune_kparams[best_idx]
    print("tuned sigma = {}".format(tuned_sigma))
    
    return tuned_sigma  
  

def test_mperceptron(dtrain,dtest,tuned_epoch):      
    acc, cm = multiclass_perceptron(dtrain,tuned_epoch,dtest)
    
    print("Accuracy = {}".format(acc))
    # print("Number of support vectors = {}".format(len_sv)) # no sv
    print("\n")
    print(cm)

def test_mkperceptron_linear(dtrain,dtest,tuned_epoch):        
    acc, len_sv, cm = multiclass_kperceptron(dtrain,tuned_epoch,dtest,linear_kernel,None)
    
    print("Accuracy = {}".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("\n")
    print(cm)

def test_mkperceptron_poly(dtrain,dtest,tuned_epoch,tuned_d):        
    kernel = polynomial_kernel
    acc, len_sv, cm = multiclass_kperceptron(dtrain,tuned_epoch,dtest,kernel,tuned_d)
    
    print("Accuracy = {}".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("\n")
    print(cm)

def test_mkperceptron_gau(dtrain,dtest,tuned_epoch,tuned_sigma):    
    kernel = gaussian_kernel
    acc, len_sv, cm = multiclass_kperceptron(dtrain,tuned_epoch,dtest,kernel,tuned_sigma)
    
    print("Accuracy = {}".format(acc))
    print("Number of support vectors = {}".format(len_sv))
    print("\n")
    print(cm)

def main():
    """Run main function."""
    # data path    
    dtrain = '../data/optdigits/train/train'
    ddev = '../data/optdigits/devel/devel'
    dtest = '../data/optdigits/test/test'

    
    # create output folder
    if not os.path.isdir('outputs'):
        os.makedirs('outputs')
    
    # tune hyperparameter T for linear kernel
    # sys.stdout = open('outputs/tune_T.txt','w')  
    # tune_epochs = list(range(1,21))   
    # tune_T(dtrain,tune_epochs,ddev)
    
    
    # tune hyperparameter d for polynomial_kernel
    # sys.stdout = open('outputs/tune_d.txt','w')
    # tuned_epoch = 13 # tuned value
    # tune_kparams = [2,3,4,5,6]
    # tune_d(dtrain,tuned_epoch,ddev,tune_kparams)

    # hyperparameter sigma for gaussian_kernel
    # sys.stdout = open('outputs/tune_sigma.txt','w')
    # tuned_epoch = 13
    # tune_kparams2 = [0.1,0.5,2,5,10]    
    # tune_sigma(dtrain,tuned_epoch,ddev,tune_kparams2)
    
    # tuned parameters
    tuned_epoch = 13
    tuned_d = 5
    tuned_sigma = 10
    
    # testing
    # sys.stdout = open('outputs/mcp.txt','w')
    # test_mperceptron(dtrain,dtest,tuned_epoch)
    
    # test linear kernel
    # sys.stdout = open('outputs/mckp_lin.txt','w')
    # test_mkperceptron_linear(dtrain,dtest,tuned_epoch)
    
    # test poly kernel
    # sys.stdout = open('outputs/mckp_poly.txt','w')
    # test_mkperceptron_poly(dtrain,dtest,tuned_epoch,tuned_d)

    
    # test gaussian kernel
    sys.stdout = open('outputs/mckp_gau.txt','w')
    test_mkperceptron_gau(dtrain,dtest,tuned_epoch,tuned_sigma)
    
   

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
