#!python
# -*- coding: utf-8 -*-#
"""
@author: Bhishan Poudel

@date: Sep 29, 2017

@email: bhishanpdl@gmail.com

:Topic: Plot data files with next/previous buttons.

"""
# Imports


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def buttons(indices, l,xs,ys,linestyles,markers,colors,titles):
    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(indices)
            l.set_xdata(xs[i])
            l.set_ydata(ys[i])
            l.set_color(colors[i])
            l.set_linestyle(linestyles[i])
            l.set_marker(markers[i])
            plt.title('{}'.format(titles[i]))
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(indices)
            l.set_xdata(xs[i])
            l.set_ydata(ys[i])
            l.set_color(colors[i])
            l.set_linestyle(linestyles[i])
            l.set_marker(markers[i])
            plt.title('{}'.format(titles[i]))
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    plt.style.use('ggplot')
    plt.show()

def read_data(infile):
    data = np.genfromtxt(infile, delimiter=None, dtype=np.double)
    return data[:, :-1 ], data[:, [-1] ]

def main():
    """Run main function."""
    df = ['dataset','devel','train','test']
    colors, markers, linestyles = list('rgbk'), list('o^o>'), list('-:-:')
    indices = range(len(df))
    titles = df

    xs = [read_data('../data/polyfit/{}.txt'.format(d))[0] for d in df]
    ys = [read_data('../data/polyfit/{}.txt'.format(d))[1] for d in df]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    l, = plt.plot(xs[0], ys[0], lw=2)
    plt.style.use('ggplot')
    plt.title('Homework 2 qn 4b')

    # plot buttons
    buttons(indices, l,xs,ys,linestyles,markers,colors,titles)


if __name__ == "__main__":
    main()
