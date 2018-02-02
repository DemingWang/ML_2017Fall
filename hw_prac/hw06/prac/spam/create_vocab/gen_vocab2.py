#!python
# -*- coding: utf-8 -*-#
"""
Create vocab file

@author: Bhishan Poudel

@date: 

"""
# Imports
import numpy as np

def create_vocab(file_name,min_freq):
    # sum(lst_of_lst,[]) will flatten the list on O(N**2)
    words = sum([ w for w in (sorted(list(set(line[1:].split()))) 
                              for line  in open(file_name))],[])   
    vocab = {w: words.count(w) for w in words 
             if words.count(w) >=min_freq }

    return vocab


def main():
    """Run main function."""
    vocab = create_vocab('data.txt',min_freq=2)
    print(vocab)


if __name__ == "__main__":
    main()
