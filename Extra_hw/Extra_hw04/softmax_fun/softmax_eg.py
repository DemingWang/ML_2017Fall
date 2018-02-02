#!python
"""
Author: Bhishan Poudel
Program: softmax_eg.py
Date: Oct 15, 2017
Description: Simple softmax function.
"""

import numpy
import numpy as np

def softmax(w):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    w : list of numbers

    Return
    ------
    a list of the same length as w of non-negative numbers

    Examples
    --------
    >>> softmax([0.1, 0.2])
    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])
    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])
    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    
    >>> res = softmax(np.array([0, 200, 10]))
    >>> np.sum(res)
    1.0
    >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
    True
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.,  1.])
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.])
    """
    
    e = np.exp(np.array(w) - np.max(w))
    dist = e / numpy.sum(e)
    return dist


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    # Usage
    ans = softmax([0.1, 0.2])
    print("ans = {}".format(ans))
