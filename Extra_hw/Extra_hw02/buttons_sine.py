"""
=======
Buttons
=======

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def get_ydata(i, powers, x):
    return x ** powers[i]

def buttons(powers, l,x):
    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(powers)
            ydata = get_ydata(i, powers, x)
            l.set_ydata(ydata)
            plt.title('power = %d'%powers[i])
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(powers)
            ydata = get_ydata(i, powers, x)
            l.set_ydata(ydata)
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

def read_data(infile):
    data = np.genfromtxt(infile, delimiter=None, dtype=np.double)
    return data[:, :-1 ], data[:, [-1] ]


def myplot(X, t,data_file,style):
    # matplotlib customization
    plt.style.use('ggplot')
    fig, ax = plt.subplots()


    # plot with label, title
    ax.plot(X,t,style,label=data_file)

    # set xlabel and ylabel to AxisObject
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Polynomial ' + data_file + ' data')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('../images/hw02qn4_'+ data_file+'.png')
    plt.show()
    plt.close()

def plot_alldata():
    data_files = ['dataset','devel','train','test']
    styles = ['ro','g^','bo','k>']
    for i, data_file in enumerate(data_files):
        X, t = read_data('../data/polyfit/{}.txt'.format(data_file))
        myplot(X,t,data_file,styles[i])


def main():
    """Run main function."""
    powers = np.arange(1, 9)
    x = np.arange(-10, 10, 0.001)
    y = x ** powers[0]


    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    l, = plt.plot(x, y, lw=2)
    plt.axhline(ls='--', c='r')
    plt.axvline(ls='--',c='r')
    plt.title('Powers of x')
    # buttons(powers, l,x)

    plot_alldata()

if __name__ == "__main__":
    main()
