#!python
# -*- coding: utf-8 -*-#
"""
:Topic: calculate this.
@author: Bhishan Poudel

@date: Sep 22, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np


def main():
    """Run main function."""
    # data    = np.arange(20).reshape((5,4))
    # col0    = data[:, [0] ]
    # col0_1  = data[:, [0,1]]
    # col0_1a = data[:, :2]
    # not_col0 = data[:, 1:]
    data     = np.arange(20).reshape((5,4))
    col0     = data[:, [0] ]
    col0a     = data[:, 0 ]
    col0_1   = data[:, [0,1]]
    col0_1a  = data[:, :2]
    not_col0 = data[:, 1:]
    last_column_array = data[:, -1]
    last_column_vec = data[:, [-1]]
    not_last = data[:,:-1]

    print("\ndata = \n{}".format(data))
    print("\ncol0 = \n{}".format(col0))
    print("\ncol0a = \n{}".format(col0a))
    print("\ncol0_1 = \n{}".format(col0_1))
    print('\ncol0_1a = ', col0_1a)
    print("\nrest = {}".format(not_col0))
    print("\nnot_last = {}".format(not_last))
    print("\nlast column as row = {}".format(last_column_array))
    print("\nlast column as column = {}".format(last_column_vec))
    print("\ndata = \n{}".format(data))

if __name__ == "__main__":
    main()
