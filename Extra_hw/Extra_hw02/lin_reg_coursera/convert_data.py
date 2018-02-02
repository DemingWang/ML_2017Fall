#!python
# -*- coding: utf-8 -*-#
"""
:Topic:
@author: Bhishan Poudel

@date:

@email: bhishanpdl@gmail.com

"""
# Imports
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{:,.4g} ".format(x)})

dat = np.loadtxt('ex1data2_orig.txt',delimiter=',')
np.savetxt('ex1data2.txt', dat,delimiter='\t')
