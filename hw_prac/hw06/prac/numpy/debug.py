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

def eg1():
    # get labels
    cmd1 = 'cut -c 1 < spam_train.txt > label1.txt'
    cmd2 = 'cut -c 1 < dense.txt > label2.txt'
    subprocess.call(cmd1, shell=True)
    subprocess.call(cmd2, shell=True)
    
    # compare labels
    l1 = np.loadtxt('label1.txt')
    l2 = np.loadtxt('label2.txt')
    for i,j in zip(l1,l2):
        print(i-j)

def main():
    """Run main function."""
    eg1()

if __name__ == "__main__":
    main()
