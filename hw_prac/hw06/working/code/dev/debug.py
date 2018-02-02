#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm

@author: Bhishan Poudel

@date: Nov 7, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import collections
import numpy as np
import subprocess

def debug1():
    # check = 'train'
    check = 'test'
    # get labels
    cmd1 = 'cut -c 1 < ../data/spam/spam_{}.txt > ../data/spam/label1.txt'.format(check)
    cmd2 = 'cut -c 1 < ../data/spam/spam_{}_dense.txt > ../data/spam/label2.txt'.format(check)
    subprocess.call(cmd1, shell=True)
    subprocess.call(cmd2, shell=True)
    
    # compare labels
    l1 = np.loadtxt('../data/spam/label1.txt')
    l2 = np.loadtxt('../data/spam/label2.txt')
    for i,j in zip(l1,l2):
        print(i-j)

def main():
    """Run main function."""
    eg1()

if __name__ == "__main__":
    main()
