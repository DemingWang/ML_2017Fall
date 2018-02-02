"""
=======
Buttons
=======

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def buttons(powers, l,x,ys):
    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(powers)
            l.set_ydata(ys[i])
            plt.title('index = %d'%powers[i])
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(powers)
            l.set_ydata(ys[i])
            plt.title('power = %d'%powers[i])
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    plt.show()

def main():
    """Run main function."""
    powers = [0,1]
    x = np.arange(5)
    ys = [np.arange(5)*2, np.arange(5)*3]


    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    l, = plt.plot(x, ys[0], lw=2)
    plt.title('Powers of x')

    # plot buttons
    buttons(powers, l,x,ys)


if __name__ == "__main__":
    main()
