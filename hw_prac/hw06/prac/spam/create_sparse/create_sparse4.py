#!python
# -*- coding: utf-8 -*-#
"""
Generate sparse data.

"""
# Imports
import collections
import numpy as np

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

            sparse_line = line[0:2] + " " + " ".join(nums) + "\n"
            print("Writing sparse matrix line: {}".format(i+1))
            fo.write(sparse_line)

def main():
    # datafiles
    fdata, min_freq  = 'data.txt', 2  
    fsparse    = 'sparse.txt'     
    fvocab     = "vocab.txt"      
    fdense     = 'dense.txt'
    
    create_sparse(fdata,fvocab,fsparse)

    

if __name__ == "__main__":
    main()
