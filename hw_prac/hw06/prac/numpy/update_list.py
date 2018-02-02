#!python
# -*- coding: utf-8 -*-#
"""
Spam Exercise (Qn 2)

@author: Bhishan Poudel

@date: Nov 9, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import collections
import numpy as np

def eg1():

    lvocab = 4
    label = 1
    indices = [1,2,3]
       
    row = [0]* (lvocab+1)
    row[0] = label
    # for idx in indices:
    #     row[idx] = 1
      
    row2 = [ 1 if i in indices else row[i] for i in range(len(row))]
    
    
    # l = [1, 2, 3, 4, 5] 
    # s = ['yes' if v == 1 
    #      else 'no' if v == 2 
    #      else 'idle' 
    #      for v in l]
            
    print("row = {}".format(row)) 
    print("row2 = {}".format(row2)) 
    
    #  0  1  2  3  4  5
    # [1, 0, 0, 0, 0]
    # [1, 1, 1, 1, 0]
    
    # example make all zeors 1
    # lst = [a if a else 2 for a in [0,1,0,3]]
    # print("lst = {}".format(lst))
    
    


def main():
    eg1()

if __name__ == "__main__":
    main()
