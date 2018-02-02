#!python
# -*- coding: utf-8 -*-#
"""
Natural Language Processing using perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 12, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np
import os

from perceptron import perceptron_train
from perceptron import aperceptron_train
from perceptron import perceptron_test

from perceptron import read_examples

from perceptron import kperceptron_train
from perceptron import quadratic_kernel
from perceptron import kperceptron_test

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
            
        print("File written: {}".format(fdense))

def run_perceptron(fdense_tr,fdense_ts,trainer,epochs,fout,model_name):
    X_train, Y_train = read_examples(fdense_tr)
    X_test,  Y_test = read_examples(fdense_ts)

    w, final_iter, mistakes = trainer(X_train, Y_train, epochs,verbose=0)
    score = perceptron_test(w, X_test)

    correct  = np.sum(score == Y_test)
    accuracy = correct/ len(score) * 100
    
    with open(fout,'w') as fo:
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        print("{} Accuracy = {:.2f} % ({} out of {} correct)".format(
              model_name, accuracy, correct, len(score)),file=fo)

def run_kperceptron(fdense_tr,fdense_ts,epochs,fout,model_name):
  X_train, y_train = read_examples(fdense_tr)
  X_test,  y_test = read_examples(fdense_ts)
  kernel = quadratic_kernel
  
  alpha, sv, sv_y,final_iter = kperceptron_train(X_train,y_train,epochs,kernel,verbose=0)  
  score = kperceptron_test(X_test,kernel,alpha,sv,sv_y)
  
  correct = np.sum(score == y_test)
  accuracy = correct/ len(score) * 100
  
  with open(fout,'w') as fo:
      print("Final iteration = {}".format(final_iter), file=fo)
      print("Total mistakes = {}".format(mistakes), file=fo)
      print("{} Accuracy = {:.2f} % ({} out of {} correct)".format(
            model_name, accuracy, correct, len(score)),file=fo)


def main():

    # first create dense matrix
    # path
    pth          = '../data/newsgroups/'
    fvocab       = pth + "newsgroups_vocab.txt"

    # sparse data
    fsparse_tr1 = pth + 'newsgroups_train1.txt'
    fsparse_ts1 = pth + 'newsgroups_test1.txt'
    fsparse_tr2 = pth + 'newsgroups_train2.txt'
    fsparse_ts2 = pth + 'newsgroups_test2.txt'

    # dense data (input for perceptron)
    fdense_tr1  = pth + 'newsgroups_train_dense1.txt'
    fdense_ts1  = pth + 'newsgroups_test_dense1.txt'
    fdense_tr2  = pth + 'newsgroups_train_dense2.txt'
    fdense_ts2  = pth + 'newsgroups_test_dense2.txt'
    
    # output files
    if not os.path.isdir('outputs'):
        os.makedirs('outputs')
        
    o        = 'outputs/'
    fout_p1  = o + 'newsgroups_model_p1.txt'
    fout_p2  = o + 'newsgroups_model_p2.txt'
    fout_ap1 = o + 'newsgroups_model_ap1.txt'
    fout_ap2 = o + 'newsgroups_model_ap2.txt'
    fout_kp1 = o + 'newsgroups_model_kp1.txt'
    fout_kp2 = o + 'newsgroups_model_kp2.txt'
    
    # model names
    p1 = 'Perceptron version 1'
    p2 = 'Perceptron version 2'
    ap1 = 'Averaged Perceptron version 1'
    ap2 = 'Averaged Perceptron version 2'
    kp1 = 'Kernel Perceptron version 1'
    kp2 = 'Kernel Perceptron version 2'
    
    ## create dense matrix as input for perceptron
    # create_dense(fsparse_tr1,fvocab, fdense_tr1)
    # create_dense(fsparse_tr2,fvocab, fdense_tr2)
    # create_dense(fsparse_ts1, fvocab, fdense_ts1)
    # create_dense(fsparse_ts2, fvocab, fdense_ts2)
    
    
    # vanilla, average, and kernel perceptrons
    epochs = 10000
    
    # perceptron version 1 and 2
    # trainer = perceptron_train
    # run_perceptron(fdense_tr1,fdense_ts1,trainer,epochs,fout_p1, p1)
    # run_perceptron(fdense_tr2,fdense_ts2,trainer,epochs,fout_p2, p2)
    # 
    # # averaged perceptron version 1 and 2
    # trainer = aperceptron_train
    # run_perceptron(fdense_tr1,fdense_ts1,trainer,epochs,fout_ap1, ap1)
    # run_perceptron(fdense_tr2,fdense_ts2,trainer,epochs,fout_ap2, ap2)

    # kernel perceptron version 1 and 2
    trainer = kperceptron_train
    run_kperceptron(fdense_tr1,fdense_ts1,epochs,fout_kp1, kp1)
    run_kperceptron(fdense_tr2,fdense_ts2,epochs,fout_kp2, kp2)


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
