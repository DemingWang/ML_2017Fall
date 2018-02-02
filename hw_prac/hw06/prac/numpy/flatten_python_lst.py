#!python
# -*- coding: utf-8 -*-#
"""
Flatten a python lists

@author: Bhishan Poudel

@date: Nov 15, 2017

"""
# Imports
import numpy as np
import itertools

def eg1():
    l = [[1, 2, 3], [4, 5, 6], [7], [8, 9]] * 1
    
    # fastest method
    # O(N)
    l2 = [item for sublist in l for item in sublist]
    
    # 10 times slower
    l2 = np.concatenate(l).tolist() # shorter
    l2 = np.concatenate(l).ravel().tolist()
    
    # fastest but long to write
    l2 = list(itertools.chain.from_iterable(l))
    
    # easiest to write but slow for large list
    # O(N**2)
    l2 = sum(l, [])
    
    print("l = {}".format(l2))

def main():
    """Run main function."""
    eg1()

if __name__ == "__main__":
    main()
