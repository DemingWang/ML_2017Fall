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
    raw_data = loadmat('data/ex6data1.mat')  
    # print(raw_data)
    return raw_data

def plot1(raw_data):
    data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  
    data['y'] = raw_data['y']

    positive = data[data['y'].isin([1])]  
    negative = data[data['y'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))  
    ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')  
    ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')  
    ax.legend()
    plt.show()
    plt.close()    

def svm_train(data):
    svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000,random_state=100)
        
    svc.fit(data[['X1', 'X2']], data['y'])  
    s = svc.score(data[['X1', 'X2']], data['y'])
    print("s = {}".format(s))
    
    # increase C
    svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000,random_state=100)  
    svc2.fit(data[['X1', 'X2']], data['y'])  
    s2 = svc2.score(data[['X1', 'X2']], data['y'])
    print("s2 = {}".format(s2))

def plot_svm_dec(data):
    svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000,random_state=100)
    svc.fit(data[['X1', 'X2']], data['y'])
    
    data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
    

    fig, ax = plt.subplots(figsize=(12,8))  
    ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')  
    ax.set_title('SVM (C=1) Decision Confidence')
    plt.show()
    plt.close()
def plot_svm_dec2(data):
    svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000,random_state=100)
    svc2.fit(data[['X1', 'X2']], data['y'])
    
    data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])
    

    fig, ax = plt.subplots(figsize=(12,8))  
    ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')  
    ax.set_title('SVM (C=100) Decision Confidence')
    plt.show()
    plt.close()

def gaussian_kernel(x1, x2, sigma):  
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))



def main():
    """Run main function."""
    # raw data
    raw_data = read_data()
    # plot1(raw_data)
    
    # data
    data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  
    data['y'] = raw_data['y']
    
    # svm_train(data)
    # plot_svm_dec(data)
    plot_svm_dec2(data)
    

if __name__ == "__main__":
    main()
