#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 21, 2017
# Last update :
###########################################################################
"""
:Topic: Numpy einnsum Examples

:Ref: https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
"""
# Imports
import numpy as np

def main():
    """Run main function."""

    A = np.arange(10) # 1d array
    B = np.arange(5, 15) # 1d array
    # print(A)
    # print(B)

    # sum
    sum_A = np.sum(A)
    sum_A1 = np.einsum('i->', A)
    # print(sum_A, sum_A1)

    # elementwise multiplication
    AB = A * B
    AB1 = np.einsum('i,i->i', A, B)
    # print(AB, AB1)

    # dot product
    AB = np.inner(A, B)
    AB = np.dot(A, B)
    AB = A @ B
    AB1 = np.einsum('i,i', A,B)
    # print(AB, AB1)

    # dot product
    A = np.array([[1, 1, 1],
           [2, 2, 2],
           [5, 5, 5]])

    B = np.array([[0, 1, 0],
           [1, 1, 0],
           [1, 1, 1]])

    AB = A @ B
    AB1 = np.einsum('ij,jk->ik', A, B)
    # print(AB)
    # print(AB1)

    # Understand einsum
    A=np.arange(6).reshape(2,3)
    B=np.arange(12).reshape(3,4)
    print("A.shape = ", A.shape)
    # print("B.shape = ", B.shape)

    # A is 2,3   and B is 3,4
    # C will be 2,4 if we do dot product A @ B
    # But we can also make 4,2 by (A@B).T or use einsum
    # i = first dim of A = last dim of C
    # j = summed over and consumed
    # k = last dim of B
    # A is ij B is jk and C required ki
    #      23      34                42
    C42 = np.einsum('ij,jk->ki',A,B)
    # print(C42)

    # trace (sum of diagonal elements)
    A = np.array([[1,2,3],
                  [3,4,5],
                  [4,5,6]])
    print(A)
    trace_A = np.einsum('ii', A)
    # trace_A = np.trace(A)
    print(trace_A)


if __name__ == "__main__":
    main()
