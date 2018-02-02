#!python
# -*- coding: utf-8 -*-#
###########################################################################
# Author      : Bhishan Poudel; Physics Graduate Student, Ohio University
# Date        : Sep 21, 2017
# Last update :
###########################################################################
"""
:Topic: Numpy linalg lstsq Example

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Run main function."""
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])
    A = np.vstack([x, np.ones(len(x))]).T
    # print(A)
    m, c = np.linalg.lstsq(A, y)[0]
    print(m,c)

    # Plot
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, m*x + c, 'r', label='Fitted line')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
