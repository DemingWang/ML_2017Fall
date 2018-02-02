#!python
# -*- coding: utf-8 -*-#
"""
Convert binary file to csv.

@author: Bhishan Poudel

@date: Oct 15, 2017

@email: bhishanpdl@gmail.com

Ref: http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/bonus/reading_mnist.ipynb
"""
# Imports
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
 
def load_mnist(path, which='train'):
 
    if which == 'train':
        labels_path = os.path.join(path, 'train-labels-idx1-ubyte')
        images_path = os.path.join(path, 'train-images-idx3-ubyte')
    elif which == 'test':
        labels_path = os.path.join(path, 't10k-labels-idx1-ubyte')
        images_path = os.path.join(path, 't10k-images-idx3-ubyte')
    else:
        raise AttributeError('`which` must be "train" or "test"')
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, n, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
 
    return images, labels

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()

def save_data(X, y):
    np.savetxt('data/train_img.csv', X[:3000, :], delimiter=',', fmt='%i')
    np.savetxt('data/train_labels.csv', y[:3000], delimiter=',', fmt='%i')

    X = np.genfromtxt('data/train_img.csv', delimiter=',', dtype=int)
    y = np.genfromtxt('data/train_labels.csv', delimiter=',', dtype=int)
    
    print("X.shape = {}".format(X.shape))
    print("y.shape = {}".format(y.shape))


def main():
    X, y = load_mnist(path='data/', which='train')
    print('Labels: %d' % y.shape[0])
    print('Rows: %d, columns: %d' % (X.shape[0], X.shape[1]))
    
    # for i in range(4):
    #     plot_digit(X, y, i)
    
    save_data(X, y)

if __name__ == "__main__":
    main()
