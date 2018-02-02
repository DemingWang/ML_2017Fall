#!python
# -*- coding: utf-8 -*-#
"""
Error analysis

@author: Bhishan Poudel

@date: Nov 13, 2017

https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
"""
# Imports
import numpy as np
from sklearn.metrics import confusion_matrix as skm_cm
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from nltk import ConfusionMatrix

from show_confusion_matrix import show_confusion_matrix

def eg1():
    y_actu = [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    cm = skm_cm(y_actu, y_pred)
    # print(cm)
    print(ConfusionMatrix(list(y_actu), list(y_pred)))
    
    # show_confusion_matrix(cm, ['Class 0', 'Class 1'])
    
    
    return cm

def eg2():
    y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
    y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred) # simple
    
    # better looking
    # df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    # print(df_confusion)
    
    # normalized
    # df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    # print(df_conf_norm)
    
    # plot
    plot_confusion_matrix(df_confusion)
 
def eg3():
    from pandas_ml import ConfusionMatrix
    y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    cm = ConfusionMatrix(y_actu, y_pred)
    cm.print_stats()
 
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()
    plt.close()

def plot_confusion_matrix1(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()

def egp1():
    cm = np.array([[13,  0,  0],[ 0, 10,  6],[ 0,  0,  9]])
    plot_confusion_matrix1(cm, ['A', 'B', 'C'])
    

def eg5():
    
    arr = [[13,1,1,0,2,0],
         [3,9,6,0,1,0],
         [0,0,16,2,0,0],
         [0,0,0,13,0,0],
         [0,0,0,0,15,0],
         [0,0,1,0,0,15]]        
    df_cm = pd.DataFrame(arr, list(range(len(arr))),
                      list(range(len(arr))))
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    plt.show()
    plt.close()
    

def main():
    """Run main function."""
    eg1()

if __name__ == "__main__":
    main()
    
"""
Table of confusion

            actual class
predicted   TP  FP
            FN  TN
            
            
F1 = 2*TP / (2*TP + FP + FN)
"""
