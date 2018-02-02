#!python
# -*- coding: utf-8 -*-#
"""
Softmax Regression for MNIST data.

@author: Bhishan Poudel

@date: Oct 15, 2017

@email: bhishanpdl@gmail.com

Ref: http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression
"""
# Imports
from tensorflow.examples.tutorials.mnist import input_data
    
def main():
    """Run main function."""
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    

if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    #  Run the main program
    main()

    # Print the time taken
    program_end_time = time.time()
    end_ctime        = time.ctime()
    seconds          = program_end_time - program_begin_time
    m, s             = divmod(seconds, 60)
    h, m             = divmod(m, 60)
    d, h             = divmod(h, 24)
    print("\nBegin time: ", begin_ctime)
    print("End   time: ", end_ctime, "\n")
    print("Time taken: {0: .0f} days, {1: .0f} hours, \
      {2: .0f} minutes, {3: f} seconds.".format(d, h, m, s))
