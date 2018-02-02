#!python
# -*- coding: utf-8 -*-#
"""


@author: Bhishan Poudel

@date: 

"""
# Imports

from sklearn.svm import SVC

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix, classification_report

import os
import numpy as np
np.set_printoptions(3)

def svm_train(fsparse_train, fsparse_test,model_name,fo):
    X_train, y_train = load_svmlight_file(fsparse_train)
    X_test, y_test = load_svmlight_file(fsparse_test)

    model = SVC(kernel='linear', random_state=0, C=5.0)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    correct = np.sum(y_test == y_pred)
    accuracy = correct / len(y_test)
    print("\n{}: accuracy = {}".format(model_name,accuracy),file=fo)
    cm = confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test,y_pred)
    print("Confusion Matrix".format(),file=fo)
    print(cm,file=fo)
    print()
    print("Reports".format())
    print(cr)


def main():
    """Run main function."""

    fsparse_train_spam = '../data/spam/spam_train_svm.txt'
    fsparse_test_spam = '../data/spam/spam_test_svm.txt'

    fsparse_train_news = '../data/newsgroups/newsgroups_train1.txt'
    fsparse_test_news = '../data/newsgroups/newsgroups_test1.txt'

    fsparse_train_news2 = '../data/newsgroups/newsgroups_train2.txt'
    fsparse_test_news2 = '../data/newsgroups/newsgroups_test2.txt'
    
    if not os.path.isdir('outputs'):
        os.makedirs('outputs')


    fo1 = open('outputs/qn2_outputs.txt','w')
    fo = open('outputs/qn2_outputs.txt','a')
    svm_train(fsparse_train_spam, fsparse_test_spam,'spam', fo)
    svm_train(fsparse_train_news, fsparse_test_news,'news1', fo)
    svm_train(fsparse_train_news2, fsparse_test_news2,'news2', fo)
    fo1.close()
    fo.close()

if __name__ == "__main__":
    main()
