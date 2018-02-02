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
        print("Sparse file is written: {}".format(fsparse))


def main():
    fdata = '../data/data_01.txt'
    fvocab = '../data/vocab_01.txt'
    fsparse = '../data/sparse_01.txt'
    create_sparse(fdata,fvocab,fsparse)


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
