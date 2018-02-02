#!python
# -*- coding: utf-8 -*-#
"""
Spam Exercise (Qn 2)

@author: Bhishan Poudel

@date: Nov 9, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import collections
import numpy as np

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

def create_dense(fsparse, fdense,fvocab):
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
    


def main():
    # datafiles
    # fdata, min_freq      = 'data.txt', 2
    
    fdata, min_freq = 'spam_train.txt', 30

    
    fsparse    = 'sparse.txt'     
    fvocab     = "vocab.txt"      
    fdense     = 'dense.txt'
    
    # create_vocab(fdata,min_freq, fvocab)
    # create_sparse(fdata,fvocab,fsparse)
    # create_dense(fsparse, fdense,fvocab)
    
    # compare labels
    l1 = np.loadtxt('label1.txt')
    l2 = np.loadtxt('label2.txt')
    for i,j in zip(l1,l2):
        print(i-j)

if __name__ == "__main__":
    main()
