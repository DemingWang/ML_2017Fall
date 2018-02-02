#!python
# -*- coding: utf-8 -*-#
"""
Error analysis

@author: Bhishan Poudel

@date: Nov 13, 2017

"""
# Imports
import numpy as np

def eg1():
    y      = np.array([1,0,0,0,0,0,1,1,1,1])
    y_pred = np.array([1,0,0,1,1,1,0,0,0,0])
    
    tp = np.sum(y[y==1]==y_pred[y_pred==1])
    print(("tp = {}".format(tp)))
    
    TP = 1 # 1,1  true positive or hit
    TN = 2 # 0,0 true negative or correct rejection
    
    FP = 3 # 0,1 FALSE ALARM or type I error
    FN = 4 # 1,0 miss or type II error
    
    P = TP + FN # condition positive
    N = TN + FP # condition negatives
    
    
    TPR      = TP / (TP + FN) # = TP / P # sensitivity,recall,hit rate
    TNR      = TN/ (TN+ FP) # specificity, true neg rate
    
    # positive predictive value
    precision = TP / (TP + FP) # PPV = TP / P
    recall = TPR
    neg_prd_v = TN / (TN + FN) # NPV
    
    Accuracy = (TP+TN) / (TP+TN+FP+FN)
    F = 2 * (precision*recall) / (precision + recall)
    
    print(("F = {}".format(F)))
    
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
