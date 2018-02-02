#!python
# -*- coding: utf-8 -*-#
"""
scipy sparse matrix

@author: Bhishan Poudel

@date: Nov 8, 2017

@email: bhishanpdl@gmail.com

https://stackoverflow.com/questions/26889597/read-sparse-matrix-in-python
"""
# Imports
import scipy as sp
import numpy as np
from scipy import sparse

def sp_sparse1():
    i=[0,5,0,1,2,1]
    j=[0,1,2,0,3,4]
    k=[1,2,3,4,5,6]
    A=sparse.csr_matrix((  k,  (i,j)   ))
    print(A)
    # print("A[0] = \n{}".format(A[0]))
    # print("\n")
    # print("A[0,2] = {}".format(A[0,0]))
    # 
    # print("A.tocoo().row = {}".format(A.tocoo().row))
    # print("A.tocoo().col = {}".format(A.tocoo().col))
    
    print("A.todok().keys() = {}".format(A.todok().keys()))
   
def sp_sparse_rand():
    from scipy.sparse import rand

    A = rand(10, 10, format='csr')
    
    print("type(A) = {}".format(type(A)))
    print("A = {}".format(A))
    
def eg3():
    # Accessing row,col such that data is non-zero
    import numpy as np
    from scipy.sparse.csc import csc_matrix

    row = np.array( [0, 1, 3])
    col = np.array( [0, 2, 2])
    data = np.array([1, 0, 16])
    # A = csc_matrix((data, (row, col)), shape=(4, 4))
    A = csc_matrix((data, (row, col)))
    
    # print("A = \n{}".format(A))
    # print("A.todense() = \n{}".format(A.todense()))
    # print("A.shape = {}".format(A.shape))
    
    # print("A[0:] = \n{}".format(A[0:]))
    # print("A[1:] = \n{}".format(A[1:]))
    # print("A[3:] = \n{}".format(A[3:]))
    
    # print("A[0:].toarray() = \n{}".format(A[0:].toarray()))
    # print("A.todense() = \n{}".format(A.todense()))
    
    # print("A[0,0] = {}".format(A[0,0]))
    # print("A[1,2] = {}".format(A[1,2]))
    # print("A[3,2] = {}".format(A[3,2]))
    
    # access the indices poniting to non-zero data
    rows, cols = A.nonzero()
    # print("rows = {}".format(rows))
    # print("cols = {}".format(cols))

    # non zero data list
    # l = [ ((i, j), A[i,j]) for i, j in zip(*A.nonzero())]
    l = [ (i, j) for i, j in zip(*A.nonzero())]
    # print("l = {}".format(l))


def main():
    """Run main function."""
    # sp_sparse1()
    # sp_sparse_rand()
    eg3()

if __name__ == "__main__":
    main()
