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
            
            # using for loop
            # for idx in indices:
            #     row[idx] = 1
            
            # use listcomps
            row = [ 1 if i in indices else row[i] for i in range(len(row))]
            
            l = " ".join(map(str,row)) + "\n"
            fo.write(l)
            
            print('Writing dense matrix line: ', i+1)
            # print("\nwords = {}".format(words))
            # print("label = {}".format(label))
            # print("idx   = {}".format(idx))
            # print("row = {}".format(row))
    
def run_create_dense():
    

def run_perceptron():
    # original data
    p          = '../data/spam/'
    fdata_tr   = p + 'spam_train.txt'       # 4000 examples rows
    fdata_ts   = p + 'spam_test.txt'        # 1000 examples rows
    
    # sparse data
    fsparse_tr = p + 'spam_train_svm.txt'   # 4000 lines
    fsparse_ts = p + 'spam_test_svm.txt'    # 1000 lines
    
    # feature matrix (vocab) ONLY FROM TRAIN!!!
    fvocab     = p + "spam_vocab.txt"       # 2376 lines features
    
    # dense data (input for perceptron)
    fdense_tr  = p + 'spam_train_dense.txt' # 4000, 2376+1 matrix
    fdense_ts  = p + 'spam_test_dense.txt'  # 1000, 2376+1 matrix
    
    # min freq to create vocab
    min_freq   = 30
    
    #===========================================================
    # # create vocab from train
    # create_vocab(fdata_tr,min_freq, fvocab)
    # 
    # # train
    # create_sparse(fdata_tr, fvocab,fsparse_tr)
    # create_dense(fsparse_tr,fvocab, fdense_tr)
    # 
    # # test
    # create_sparse(fdata_ts,fvocab,fsparse_ts)
    # create_dense(fsparse_ts, fvocab, fdense_ts)
    
    #===========================================================
    ## Train and test vanilla perceptron
    epochs = 200
    X_train, Y_train = read_examples(fdense_tr)
    X_test,  Y_test = read_examples(fdense_ts)
    
    # print("X_train.shape = {}".format(X_train.shape)) # (4000, 2376)
    # print("Y_train.shape = {}".format(Y_train.shape)) # (4000,)
    # print("X_test.shape = {}".format(X_test.shape)) # (1000, 2376)
    # print("Y_test.shape = {}".format(Y_test.shape)) # (1000,)
    
    
    w, final_iter, mistakes = perceptron_train(X_train, Y_train, epochs,verbose=0)
    score = perceptron_test(w, X_test)
    
    correct  = np.sum(score == Y_test)
    accuracy = correct/ len(score) * 100
    
    with open('outputs/spam_model_p.txt','w') as fo:
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        print("Final parameter vector w = {}".format(w), file=fo)
        print("Vanilla Perceptron Accuracy = {:.2f} % ({} out of {} correct)".format(
              accuracy, correct, len(score)), file=fo)

def run_aperceptron():
    # original data
    p          = '../data/spam/'
    fdata_tr   = p + 'spam_train.txt'       # 4000 examples rows
    fdata_ts   = p + 'spam_test.txt'        # 1000 examples rows
    
    # sparse data
    fsparse_tr = p + 'spam_train_svm.txt'   # 4000 lines
    fsparse_ts = p + 'spam_test_svm.txt'    # 1000 lines
    
    # feature matrix (vocab) ONLY FROM TRAIN!!!
    fvocab     = p + "spam_vocab.txt"       # 2376 lines features
    
    # dense data (input for perceptron)
    fdense_tr  = p + 'spam_train_dense.txt' # 4000, 2376+1 matrix
    fdense_ts  = p + 'spam_test_dense.txt'  # 1000, 2376+1 matrix
    
    # min freq to create vocab
    min_freq   = 30
    
    #===========================================================
    # # create vocab from train
    # create_vocab(fdata_tr,min_freq, fvocab)
    # 
    # # train
    # create_sparse(fdata_tr, fvocab,fsparse_tr)
    # create_dense(fsparse_tr,fvocab, fdense_tr)
    # 
    # # test
    # create_sparse(fdata_ts,fvocab,fsparse_ts)
    # create_dense(fsparse_ts, fvocab, fdense_ts)
    
    
    #===========================================================
    ## Train and test averaged perceptron
    epochs = 200
    X_train, Y_train = read_examples(fdense_tr)
    X_test,  Y_test = read_examples(fdense_ts)
    
    
    w, final_iter,mistakes = aperceptron_train(X_train, Y_train, epochs)
    
    score = perceptron_test(w, X_test)
    correct = np.sum(score == Y_test)
    accuracy = correct/ len(score) * 100
    
    with open('outputs/spam_model_ap.txt', 'w') as fo:
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        print("Final parameter vector w = {}".format(w), file=fo)
        print("Vanilla Perceptron Accuracy = {:.2f} % ({} out of {} correct)".format(
              accuracy, correct, len(score)), file=fo)

def run_kperceptron():
    data_tr = '../data/spam/spam_train_dense.txt'
    data_ts = '../data/spam/spam_test_dense.txt'
    
    #===========================================================
    ## kernel perceptron    
    epochs = 200
    X_train, y_train = read_examples(data_tr)
    X_test,  y_test = read_examples(data_ts)
    kernel = quadratic_kernel
    
    alpha, sv, sv_y,final_iter = kperceptron_train(X_train,y_train,epochs,kernel,verbose=0)  
    score = kperceptron_test(X_test,kernel,alpha,sv,sv_y)
    
    correct = np.sum(score == y_test)
    accuracy = correct/ len(score) * 100
    
    with open('outputs/spam_model_kp.txt','w') as fo:
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Kernel Perceptron Accuracy = {:.2f} % ({} out of {} correct)".format(
              accuracy, correct, len(score)),file=fo)



def main():
    run_perceptron()
    # run_aperceptron()
    # run_kperceptron()    
    
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
    
