#!python
# -*- coding: utf-8 -*-#
"""
SVM auto_examples

@author: Bhishan Poudel

@date: Nov 19, 2017
http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-6/
"""
# Imports
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sb  
from scipy.io import loadmat
from sklearn import svm

def read_data():
    raw_data = loadmat('data/ex6data2.mat')  
    # print(raw_data)
    return raw_data

def plot1(data):
    positive = data[data['y'].isin([1])]  
    negative = data[data['y'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))  
    ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')  
    ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')  
    ax.legend()
    plt.show()
    plt.close()    

def gaussian_kernel(x1, x2, sigma):  
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

def main():
    """Run main function."""
    # raw data
    raw_data = read_data()
    plot1(raw_data)
    
    # data
    data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  
    data['y'] = raw_data['y']
    
    # svm_train(data)
    # plot_svm_dec(data)
    # plot_svm_dec2(data)
    

if __name__ == "__main__":
    main()
