#!python
# -*- coding: utf-8 -*-#
"""
Plot top 50 vocabularies.

@author: Bhishan Poudel

@date: Nov 21, 2017

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rc('font',**{'size':20, 'family':'fantasy'})

from collections import Counter
from matplotlib import colors as mcolors

def plot_words(fvocab):
    idx, words,counts = np.loadtxt(fvocab,dtype='bytes',unpack=True).astype(str)
    words, counts = words[0:20], counts[0:20]
    words = [i+'_'+word for (i,word) in zip(idx,words)]
    

    plt.figure(figsize=(8,8))
    c = list(mcolors.CSS4_COLORS.values())
    
    for i in range(len(words)):
        x = np.random.uniform(low=0, high=1)
        y = np.random.uniform(low=0, high=1)
        plt.text(x, y, words[i], 
                 size=np.sqrt(float(counts[i])), 
                 rotation=np.random.choice([-90, 0.0, 90]), 
                 color=c[np.random.randint(0,len(c))])
      
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    plt.show()

def plot_words2(fvocab):
    
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rc('font',**{'size':20, 'family':'fantasy'})

    from matplotlib import colors as mcolors

    words = ['an', 'apple', 'in', 'a', 'tree']
    counts = [23,   12,      45,  20,   9]
    maxcounts = max(counts)
    sat = [c / maxcounts for c in counts]

    plt.figure(figsize=(8,8))

    # convert colors to HSV
    css4_colors = list(mcolors.CSS4_COLORS.values())
    ncolors = len(words)
    rgb = [mcolors.to_rgb(c) for c in css4_colors]
    hsv = [mcolors.rgb_to_hsv(c) for c in rgb]

    for i in range(len(words)):
        x = np.random.uniform(low=0, high=1)
        y = np.random.uniform(low=0, high=1)

        # select a color
        icolor = np.random.randint(0, ncolors)
        color = hsv[icolor]
        color[1] = sat[i] # here change saturation, index 1 or brightness, index 2 according to the counts list

        plt.text(x, y, words[i], 
                 size=counts[i] * 2, 
                 rotation=np.random.choice([-90, 0.0, 90]), 
                 color=mcolors.hsv_to_rgb(color)) # don't forget to get back to rgb

    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    plt.show()

def main():
    """Run main function."""
    fvocab = '../data/vocab_m1.txt'
    plot_words2(fvocab)

if __name__ == "__main__":
    import time

    # Beginning time
    program_begin_time = time.time()
    begin_ctime        = time.ctime()

    # Run the main program
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
