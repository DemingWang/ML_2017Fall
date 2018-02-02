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
import pandas as pd

def table_plot():
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(True)
    ax.axis('off')
    ax.axis('tight')

    df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()

    plt.show()

def main():
    """Run main function."""
    table_plot()

if __name__ == "__main__":
    main()
