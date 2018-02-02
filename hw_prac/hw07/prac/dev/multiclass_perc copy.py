#!python
# -*- coding: utf-8 -*-#
"""
Multiclass perceptron

@author: Bhishan Poudel

@date: Nov 18, 2017

"""
# Imports
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import perceptron
from sklearn.metrics import confusion_matrix

from kperceptron import linear_kernel
from kperceptron import polynomial_kernel
from kperceptron import gaussian_kernel

from kperceptron import kperceptron_train
from kperceptron import kperceptron_test
from kperceptron import read_data

def read_data(fdata):
    *X,t = np.genfromtxt(fdata,unpack=True,dtype='f')
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    
    return X,t

def multiclass_perceptron(data_pth_train,epoch,data_pth_devel_test):
    """
    Training data ==>  data_pth_train+ str(i) + '.txt' 
    """
    # training data
    nclass = 10
    ftrain = data_pth_train + '1.txt'
    X_train, t_train = read_data(ftrain)
    weights = np.zeros([X_train.shape[1], nclass])
    
    # testing data or devel data
    fdata = data_pth_devel_test+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    
    # instanciate a model
    net = perceptron.Perceptron(max_iter=epoch, verbose=0, 
                                random_state=100, fit_intercept=True)
    
    # fit the model
    for i in range(nclass):
        fftrain           = data_pth_train+ str(i) + '.txt'
        X_train, t_train  = read_data(fftrain)
        
        # get weight
        net.fit(X_train,t_train)
        w = net.coef_
        
        # normalize w
        w = preprocessing.normalize(w,norm='l1')
        weights[:,i] = w

    
    ##get the prediction
    hyp    =  X.dot(weights)
    t_pred = np.argmax(hyp,axis=1)
    
    # accuracy
    correct = np.sum(t_pred==t_actu)
    accuracy = correct / len(t_actu) * 100
    # print("accuracy = {:.2f} %  ({} out of {} correct)".format(
        # accuracy,correct, len(t_actu)))
    
    # print(confusion_matrix(t_actu,t_pred))
    return accuracy

def multiclass_kperceptron(data_pth_train,epoch,data_pth_devel_test):
    # training data
    nclass = 10
    ftrain = data_pth_train + '1.txt'
    X_train, t_train = read_data(ftrain)
    
    # testing data or devel data
    fdata = data_pth_devel_test+'.txt'
    *X,t_actu = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X = np.array(X).T
    X = np.c_[np.ones(X.shape[0])[np.newaxis].T,X]
    weights = np.zeros([X.shape[0], nclass])
    
  
    # fit the model
    epochs,kernel, kparam = 2, polynomial_kernel, 3
    for i in range(nclass):
        fftrain           = data_pth_train+ str(i) + '.txt'
        X_train, t_train  = read_data(fftrain)
        
        # get weight
        alpha, sv, sv_y = kperceptron_train(X_train,t_train,epochs,kernel,kparam)
        _, w = kperceptron_test(X,kernel,kparam,alpha,sv,sv_y)

        # normalize w
        # w = preprocessing.normalize(w,norm='l1')
        weights[:,i] = w

    
    ##get the prediction
    hyp    =  weights
    t_pred = np.argmax(hyp,axis=1)
    
    # accuracy
    correct = np.sum(t_pred==t_actu)
    accuracy = correct / len(t_actu) * 100
    # print("accuracy = {:.2f} %  ({} out of {} correct)".format(
        # accuracy,correct, len(t_actu)))
    
    # print(confusion_matrix(t_actu,t_pred))
    return accuracy

def tune_T(data_pth,epochs,data_pth_devel_test):
    accuracies = []
    for e in range(1,epochs):
        accuracy = multiclass_perceptron(data_pth,e,data_pth_devel_test)
        accuracies.append(accuracy)
        print("epoch {}: accuracy = {:.2f}".format(e, accuracy))
        
    best_epoch = np.argmax(accuracies)+1
    print("best epoch = {}".format(best_epoch))
    
    return best_epoch
def tune_d(data_pth,epochs,data_pth_devel_test):
    accuracies = []
    for e in range(1,epochs):
        accuracy = multiclass_kperceptron(data_pth,e,data_pth_devel_test)
        accuracies.append(accuracy)
        print("epoch {}: accuracy = {:.2f}".format(e, accuracy))
        
    best_epoch = np.argmax(accuracies)+1
    print("best epoch = {}".format(best_epoch))
    
    return best_epoch

def check_kperceptron():
    # data file
    data_tr = '../data/optdigits/train/train0.txt'
    data_ts = '../data/optdigits/devel/devel0.txt'
    # data_tr = 'data/extra/train.txt'
    # data_ts = 'data/extra/test.txt'
    
    ## data   
    X_train, y_train = read_data(data_tr)
    X_test,  y_test = read_data(data_ts)
    
    # kernels
    # epochs,kernel, kparam = 200, gaussian_kernel, 0.5
    epochs,kernel, kparam = 200, polynomial_kernel, 3
    
    # fit the kernel perceptron
    alpha, sv, sv_y = kperceptron_train(X_train,y_train,epochs,kernel,kparam)
    
    y_pred, hyp = kperceptron_test(X_test,kernel,kparam,alpha,sv,sv_y)
    print("hyp.shape = {}".format(hyp.shape)) # (1000,)
    
    
    # correct
    correct = np.sum(y_pred == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_pred)))    

def main():
    """Run main function."""
    # data path
    data_train = '../data/optdigits/train/train.txt'
    data_devel = '../data/optdigits/train/devel.txt'
    data_test = '../data/optdigits/train/test.txt'
    
    data_pth_train = '../data/optdigits/train/train'
    data_pth_devel = '../data/optdigits/devel/devel'
    data_pth_test = '../data/optdigits/test/test'
    
    # actual target
    t_actu_train = np.loadtxt(data_train, usecols=(-1),delimiter=',')
    # tune epochs T
    epochs = 1 # tuned value

    # tune_T(data_pth_train,epochs,data_pth_devel)
    tune_d(data_pth_train,epochs,data_pth_devel)
    
    # kernel perceptron degree d
    # check_kperceptron()

if __name__ == "__main__":
    main()
