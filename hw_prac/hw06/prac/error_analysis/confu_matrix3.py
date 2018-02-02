#!python
# -*- coding: utf-8 -*-#
"""
confusion ConfusionMatrix

@author: Bhishan Poudel

@date: Nov 13, 2017
http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
"""
# Imports
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def eg1():
    
    binary = np.array([[4, 1],
                       [1, 2]])

    fig, ax = plot_confusion_matrix(conf_mat=binary)
    plt.show()

def eg2():
    multiclass = np.array([[2, 1, 0, 0],
                       [1, 2, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    fig, ax = plot_confusion_matrix(conf_mat=multiclass)
    plt.show()
    
def main():
    """Run main function."""
    eg2()

if __name__ == "__main__":
    main()
