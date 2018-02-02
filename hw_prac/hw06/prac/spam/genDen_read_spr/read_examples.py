#!python
# -*- coding: utf-8 -*-#
"""
Numpy fromregex examples

@author: Bhishan Poudel

@date:  Nov 13, 2017

"""
# Imports
import numpy as np
from scipy import sparse

def read_examples(fsparse):    
    row, col, data = [],[],[]
    labels = []
    with open(fsparse) as fi:
        for i, line in enumerate(fi):
            words = line.strip('\n').split(':')
            words = " ".join(words).split()

            label = int(words[0])
            labels.append(label)
            row.append(i); col.append(0); data.append(label)

            indices = [int(w) for (i,w) in enumerate(words) if int(i)%2]
            for j in indices:   # quick-n-dirty version
                row.append(i); col.append(j); data.append(1)
    
    # now create sparse and dense matrix
    M = sparse.coo_matrix((data,(row,col)))
    M = M.A
    data = M[:,1:]
    labels = np.array(labels)
                
    return data, labels
    
def main():
    fsparse = 'sparse.txt'
    fdense = 'dense.txt'
    data, labels = read_examples(fsparse)
    print("labels = {}".format(labels))
    print("data = \n{}".format(data))


if __name__ == "__main__":
    main()

"""
labels =  [1 0 1]
data = 


"""
