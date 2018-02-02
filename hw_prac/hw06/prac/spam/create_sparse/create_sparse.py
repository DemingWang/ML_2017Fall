#!python
# -*- coding: utf-8 -*-#
"""
Create sparse vector

@author: Bhishan Poudel

@date: Nov 20, 2017

"""
# Imports
import numpy as np

def create_sparse(fdata,fvocab,fsparse):
    index = np.loadtxt(fvocab, dtype=('bytes'), usecols=(0),unpack=True).astype(int)
    token = np.loadtxt(fvocab, dtype=('bytes'),usecols=(1), unpack=True).astype(str)
    my_dict = dict(zip(token,index))

    data = []
    counts = []
    fo= open(fsparse,'w')

    # read each line in data file
    for line in open(fdata,'r'):
        words = line.split()
        label = words[0]
        words1 = words[1:]
        words1 = list(set(words1))

        # read each words and compare with vocab_text
        counts.clear()
        for w in words1:
            if w in my_dict.keys():

                count = my_dict[w]
                counts.append(count)
        
        # add values 1 to every feature_vector
        data.clear()
        for i in counts:
            dat= str(i)+":1"
            data.append(dat)

        # print label and data
        print(label,' '," ".join(map(str,data)),file=fo)
        print(len(data))
        
    # always close files
    fo.close()
    return count

def main():
    """Run main function."""
    vocab_file = "../data/vocab_m1.txt"
    data_file ="../data/data_m1.txt"
    file_out = '../data/sparse_m1.txt'

    create_sparse(data_file,vocab_file,file_out)


if __name__ == "__main__":
    main()
