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
import matplotlib.pyplot as plt

def table_plot():
    # Table only
    fig, ax =plt.subplots()
    table_data = np.random.random((10,3))
    collabel=("col 1", "col 2", "col 3")
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=table_data,colLabels=collabel,loc='center')
    ax.plot(table_data[:,0],table_data[:,1])

    # Both table and plot
    # fig, axs =plt.subplots(2,1)
    # clust_data = np.random.random((10,3))
    # collabel=("col 1", "col 2", "col 3")
    # axs[0].axis('tight')
    # axs[0].axis('off')
    # the_table = axs[0].table(cellText=clust_data,colLabels=collabel,loc='center')
    # axs[1].plot(clust_data[:,0],clust_data[:,1])


    # show
    plt.show()

def main():
    """Run main function."""
    table_plot()

if __name__ == "__main__":
    main()
