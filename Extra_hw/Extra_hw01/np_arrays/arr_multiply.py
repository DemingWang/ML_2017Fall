#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 20, 2017
# Last update :
###########################################################################
# Ref: https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
"""
:Topic: Multiply numpy arrays

:Runtime:

"""
# Imports
import numpy as np

def matrix_mul():
    a = np.arange(1,10).reshape(3,3)
    b = np.arange(3)

    # Usual method
    # Ab = a.dot(b)

    # python 3.5
    # Ab = a@b


    # using eisensum
    # NOTE: if we use np.int8 np.eisensum may fail
    # Ab = np.einsum('ji,i->j', a, b,dtype=np.double)

    # Using numpy 1.10
    Ab = np.matmul(a, b)


    print (Ab)
    return Ab


def main():
    """Run main function."""
    Ab = matrix_mul()
    print("Ab.shape = ", Ab.shape)

if __name__ == "__main__":
    main()
