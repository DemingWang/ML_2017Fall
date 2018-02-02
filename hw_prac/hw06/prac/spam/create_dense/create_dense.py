#!python
# -*- coding: utf-8 -*-#
# Imports
import numpy as np
import collections

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

            print('Writing dense file line: ', i+1)
            # print("\nwords = {}".format(words))
            # print("label = {}".format(label))
            # print("idx   = {}".format(idx))
            # print("row = {}".format(row))
        print("Dense file written : {}".format(fdense))

def main():
    fsparse = '../data/sparse.txt'
    fvocab = '../data/vocab.txt'
    fdense = '../data/dense.txt'
    create_dense(fsparse, fvocab, fdense)


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
