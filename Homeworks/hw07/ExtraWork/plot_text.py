#!python
# -*- coding: utf-8 -*-#
"""
Plot text such that textsize proportional to value.

@author: Bhishan Poudel

@date:  Nov 29, 2017

"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',**{'size':20, 'family':'fantasy'})

from matplotlib import colors as mcolors
np.random.seed(0)


def plot_words():
    counts, words = np.genfromtxt('newsgroups_vocab.txt',unpack=True,dtype='str')
    counts = [int(c) for c in counts[0:20]][::-1]
    words = words[0:20]
    words = [w + "_" + str(c) for (w,c) in zip(words,counts[::-1])]
    
    print(counts, words)
    
    
    maxcounts = max(counts)
    sat = [c / maxcounts for c in counts]
    
    plt.figure(figsize=(8,8))
    
    # choose hsb color values (vary saturation only)
    hue = 0.6
    brightness = 1
    
    for i in range(len(words)):
        x = np.random.uniform(low=0, high=1)
        y = np.random.uniform(low=0, high=1)
    
        # select a color
        color = (hue, sat[i], brightness)
    
        plt.text(x, y, words[i],
                 size=counts[i] * 2,
                 rotation=np.random.choice([-90, 0.0, 90]),
                 color=mcolors.hsv_to_rgb(color))
    
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    plt.savefig('newsgroups_vocab.png')
    plt.close()

def main():
    plot_words()
    
    
if __name__ == "__main__":
    plot_words()