#!python
# -*- coding: utf-8 -*-#
"""
Create dense data from sparse data.

"""
# Imports
import collections
import numpy as np


def create_dense(fsparse,fvocab):
    # number of lines in vocab
    lvocab  = sum(1 for line in open(fvocab))
    dense = np.zeros((lvocab,lvocab),dtype=int)
    labels = []
    
    # create dense file
    with open(fsparse) as fi:
        for i, line in enumerate(fi):
            # get numbers from line
            nums = line.strip('\n').split(':')
            nums = " ".join(nums).split()
            
            # first number is the label
            label = int(nums[0])
            labels.append(label)
            
            # get index and greate dense row
            idx = [int(w)-1 for (i,w) in enumerate(nums) if int(i)%2]
            dense[i][idx] = np.ones(len(idx))
    return labels, dense


def main():
    # datafiles
    fsparse = 'sparse.txt'
    fvocab = 'vocab.txt'
    
    labels, dense = create_dense(fsparse,fvocab)
    print("labels = {}".format(labels))
    print("dense = \n{}".format(dense))

if __name__ == "__main__":
    main()
