#!python
# -*- coding: utf-8 -*-#
"""
Get first digits eg: -1 this is first line.

@author: Bhishan Poudel

@date: Nov 15, 2017

"""
# Imports
from itertools import groupby

def eg1():
    my_str = "-12 hi 89"
    
    l = [int(''.join(i)) for is_digit, i in groupby(my_str, str.isdigit) if is_digit]
    
    print("l = {}".format(l))
    
    
def main():
    """Run main function."""
    eg1()

if __name__ == "__main__":
    main()
