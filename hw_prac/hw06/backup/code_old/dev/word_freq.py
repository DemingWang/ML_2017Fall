#!python
# -*- coding: utf-8 -*-#
"""
Counting the words.

@author: Bhishan Poudel

@date: Nov 9, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import collections


def read_example(ifile,ofile):
    wordcount = collections.Counter()
    with open(ifile) as fi:
        for line in fi:
            wordcount.update(line[1:].split())

    pairs = [(i,j) for i,j in wordcount.items() if j>1]
    pairs = sorted(pairs, key=lambda word: word[0], reverse=False)
    
    print(pairs)
    with open(ofile,'w') as fo:
        for i,j in pairs:
            fo.write("{} {}\n".format(i,j))


def main():
    """Run main function."""
    ifile = 'data2.txt'
    ofile = 'counts.txt'
    read_example(ifile,ofile)

if __name__ == "__main__":
    main()
