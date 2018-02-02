#!python
# -*- coding: utf-8 -*-#
"""
Spam Exercise (Qn 2)

@author: Bhishan Poudel

@date: Nov 9, 2017

@email: bhishanpdl@gmail.com

count lines: wc -l file.txt
Get first char of lines of file: cut -c 1 file.txt | head
"""
# Imports
import numpy as np
import collections

from perceptron import perceptron_train
from perceptron import aperceptron_train
from perceptron import perceptron_test
from perceptron import read_examples
from perceptron import confusion_matrix

from perceptron import kperceptron_train
from perceptron import quadratic_kernel
from perceptron import kperceptron_test


def create_vocab(fdata,min_freq,fvocab):

    # count the words and frequencies
    wordcount = collections.Counter()
    with open(fdata) as fi:
        for line in fi:
            wordcount.update(set(line[1:].split()))

    pairs = [(w,f) for w,f in wordcount.items() if f>=min_freq ]

    # do not include stopwords
    # fstopwords = 'stopwords.txt'
    # stopwords = np.loadtxt(fstopwords,dtype='str')
    # pairs = [(w,f) for w,f in wordcount.items() if f>=min_freq if w not in stopwords]

    # sort alphabetically
    pairs = sorted(pairs, key=lambda word: word[0], reverse=0)

    # sort by number of occurrence
    # pairs = sorted(pairs, key=lambda word: word[1], reverse=1)

    print("len(vocab) = {}".format(len(pairs)))
    with open(fvocab,'w') as fo:
        for i in range(len(pairs)):
            fo.write("{} {}\n".format(i+1,pairs[i][0]))

            # write index token freq
            # fo.write("{} {} {}\n".format(i+1,pairs[i][0], pairs[i][1]))

def create_sparse(fdata,fvocab,fsparse):
    # read index token freq
    # idx,token,freq = np.genfromtxt(fvocab, dtype=str, unpack=True)

    # read index and token
    idx,token = np.genfromtxt(fvocab, dtype=str, unpack=True)
    d = dict(zip(token,idx))

    with open(fdata) as fi, \
         open(fsparse,'w') as fo:

        for i,line in enumerate(fi):
            nums = [ int(d[w]) for w in line[1:].split() if w in token ]
            nums = sorted(list(set(nums)))
            nums = [str(n)+":1" for n in nums ]

            sparse_line = line[0] + " " + " ".join(nums) + "\n"
            print("Writing sparse matrix line: {}".format(i+1))
            fo.write(sparse_line)

def create_dense(fsparse, fvocab, fdense):
    # number of lines in vocab
    lvocab = sum(1 for line in open(fvocab))

    # create dense file
    with open(fsparse) as fi, open(fdense,'w') as fo:
        for i, line in enumerate(fi):
            words = line.strip('\n').split(':')
            words = " ".join(words).split()

            label = int(words[0])
            indices = [int(w) for (i,w) in enumerate(words) if int(i)%2]

            row = [0]* (lvocab+1)
            row[0] = label

            # fill row elements
            row = [ 1 if i in indices else row[i] for i in range(len(row))]

            l = " ".join(map(str,row)) + "\n"
            fo.write(l)

            print('Writing dense matrix line: ', i+1)
            # print("\nwords = {}".format(words))
            # print("label = {}".format(label))
            # print("idx   = {}".format(idx))
            # print("row = {}".format(row))

def run_create_dense(fdata_tr,fsparse_tr,fdense_tr,
                     fdata_ts,fsparse_ts,fdense_ts,fvocab,min_freq):
    # create vocab from train
    create_vocab(fdata_tr,min_freq, fvocab)

    # train
    create_sparse(fdata_tr, fvocab,fsparse_tr)
    create_dense(fsparse_tr,fvocab, fdense_tr)

    # test
    create_sparse(fdata_ts,fvocab,fsparse_ts)
    create_dense(fsparse_ts, fvocab, fdense_ts)

def run_perceptron(X_train, Y_train, X_test, Y_test, epochs):
    """
    Return: outputs/spam_model_p.txt
    """
    
    # get params from x,y train
    w, final_iter, mistakes = perceptron_train(X_train, Y_train, epochs,verbose=0)
    
    # predict for test
    y_pred = perceptron_test(w, X_test)

    # get accuracy
    correct  = np.sum(y_pred == Y_test)
    acc      = correct/ len(y_pred) * 100
    
    # get confusion matrix
    cm = confusion_matrix(Y_test,y_pred)
    TN, FN, FP, TP = cm.flatten()
    
    P = TP + FN
    N = TN + FP
    
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    
    F1        = 2 * precision * recall / (precision + recall)
    
    accuracy  = (TP + TN) / (P + N)

    with open('outputs/spam_model_p.txt','w') as fo:
        print("Vanilla Perceptron Statistics",file=fo)
        print("=============================",file=fo)
        
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        
        print("\n",file=fo)
        print("Parameter w = {}".format(w), file=fo)
        print("Accuracy    = {:.2f} % ({} out of {} correct)".format(
              acc, correct, len(y_pred)), file=fo)

        print("\n",file=fo)
        print("F1-score = {:.2f}".format(F1),file=fo)
        print("Accuracy = {:.2f}".format(accuracy),file=fo)
        print("Confusion matrix is given below",file=fo)
        print("Diagonals are True values.",file=fo)
        print("       True_0 True_1",file=fo)
        print("       --------------",file=fo)
        print("Pred_0| {}      {}".format(TN,FN),file=fo)
        print("Pred_1| {}      {}".format(FP,TP),file=fo)
        
def run_aperceptron(X_train, Y_train, X_test, Y_test, epochs):
    """
    Return: outputs/spam_model_ap.txt
    """
    w, final_iter,mistakes = aperceptron_train(X_train, Y_train, epochs)

    y_pred = perceptron_test(w, X_test)
    correct = np.sum(y_pred == Y_test)
    acc     = correct/ len(y_pred) * 100

    # get confusion matrix
    cm = confusion_matrix(Y_test,y_pred)
    TN, FN, FP, TP = cm.flatten()
    
    P = TP + FN
    N = TN + FP
    
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    
    F1        = 2 * precision * recall / (precision + recall)
    
    accuracy  = (TP + TN) / (P + N)

    with open('outputs/spam_model_ap.txt','w') as fo:
        print("Averaged Perceptron Statistics",file=fo)
        print("=============================",file=fo)
        
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        
        print("\n",file=fo)
        print("Parameter w = {}".format(w), file=fo)
        print("Accuracy    = {:.2f} % ({} out of {} correct)".format(
              acc, correct, len(y_pred)), file=fo)

        print("\n",file=fo)
        print("F1-score = {:.2f}".format(F1),file=fo)
        print("Accuracy = {:.2f}".format(accuracy),file=fo)
        print("Confusion matrix is given below",file=fo)
        print("Diagonals are True values.",file=fo)
        print("       True_0 True_1",file=fo)
        print("       --------------",file=fo)
        print("Pred_0| {}      {}".format(TN,FN),file=fo)
        print("Pred_1| {}      {}".format(FP,TP),file=fo)

def run_kperceptron(X_train, Y_train, X_test, Y_test, epochs):
    """
    Return: outputs/spam_model_kp.txt
    """
    kernel = quadratic_kernel
    alpha, sv, sv_y,final_iter,mistakes = kperceptron_train(X_train,Y_train,epochs,kernel,verbose=1)
    y_pred = kperceptron_test(X_test,kernel,alpha,sv,sv_y)

    correct = np.sum(y_pred == Y_test)
    acc     = correct/ len(y_pred) * 100

    # get confusion matrix
    cm = confusion_matrix(Y_test,y_pred)
    TN, FN, FP, TP = cm.flatten()
    
    P = TP + FN
    N = TN + FP
    
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    
    F1        = 2 * precision * recall / (precision + recall)
    
    accuracy  = (TP + TN) / (P + N)

    with open('outputs/spam_model_kp.txt','w') as fo:
        print("Kernel Perceptron Statistics",file=fo)
        print("=============================",file=fo)
        
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        
        print("\n",file=fo)
        # print("Parameter w = {}".format(w), file=fo) # there are alphas
        print("Accuracy    = {:.2f} % ({} out of {} correct)".format(
              acc, correct, len(y_pred)), file=fo)

        print("\n",file=fo)
        print("F1-score = {:.2f}".format(F1),file=fo)
        print("Accuracy = {:.2f}".format(accuracy),file=fo)
        print("Confusion matrix is given below",file=fo)
        print("Diagonals are True values.",file=fo)
        print("       True_0 True_1",file=fo)
        print("       --------------",file=fo)
        print("Pred_0| {}      {}".format(TN,FN),file=fo)
        print("Pred_1| {}      {}".format(FP,TP),file=fo)

def main():
    # original data
    p          = '../data/spam/'
    fdata_tr   = p + 'spam_train.txt'
    fdata_ts   = p + 'spam_test.txt'

    # sparse data
    fsparse_tr = p + 'spam_train_svm.txt'
    fsparse_ts = p + 'spam_test_svm.txt'

    # feature matrix (vocab) ONLY FROM TRAIN!!!
    fvocab     = p + "spam_vocab.txt"

    # dense data (input for perceptron)
    fdense_tr  = p + 'spam_train_dense.txt'
    fdense_ts  = p + 'spam_test_dense.txt'

    # min freq to create vocab
    min_freq   = 30

    # variables for perceptron
    epochs = 200
    X_train, Y_train = read_examples(fdense_tr)
    X_test,  Y_test = read_examples(fdense_ts)

    # data processing
    run_create_dense(fdata_tr,fsparse_tr,fdense_tr,fdata_ts,fsparse_ts,fdense_ts,fvocab,min_freq)

    run_perceptron(X_train, Y_train, X_test, Y_test, epochs)
    run_aperceptron(X_train, Y_train, X_test, Y_test, epochs)
    run_kperceptron(X_train, Y_train, X_test, Y_test, epochs)

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
