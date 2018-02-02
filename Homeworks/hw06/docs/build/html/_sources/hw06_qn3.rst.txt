Qn 3: Atheism vs. Religion
========================================================================

In this question we will train and evaluate the binary perceptron and
avergae perceptron algorithms on a subset of the 20 newsgroups dataset.

Information:

  - 857 positive examples on newsgroups dataset
  - 570 negative examples on newsgroups dataset
  - label 1 for atheism and label 0 for religion
  - data is preprocessed and presented as sparse matrix

Detailed information is given in scikit learn website:

http://scikit-learn.org/stable/datasets/twenty_newsgroups.html


Version 1
-----------

 - it used the method TF-IDF formula to proceess the raw data
 - sparse datafiles are `newsgroups_train1.txt` and `newsgroups_test1.txt`

Version 2
-----------

  - term frequency (TF) is set to 1 and IDF is irrelevant.
  - sparse datafiles are `newsgroups_train2.txt` and `newsgroups_test2.txt`


Vanilla perceptron method
----------------------------
Here, in the directory **code/outputs/** we will create following files:

  - newsgroups_model_p1.txt for version 1
  - newsgroups_model_p2.txt for version 2


Averaged perceptron method
----------------------------
Here, in the directory **code/outputs/** we will create following files:

  - newsgroups_model_ap1.txt for version 1
  - newsgroups_model_ap2.txt for version 2

Kernel perceptron method (Extra Work)
---------------------------------------
Here, in the directory **code/outputs/** we will create following files:

  - newsgroups_model_kp1.txt for version 1
  - newsgroups_model_kp2.txt for version 2


Summary
--------

20 newsgroups atheism vs religion classification::

  ============  ======================  ====================== 
  Perceptron    Version 1 Accuracy %    Version 2 Accuracy %
  ============  ======================  ====================== 
  Vanilla       59.12                   58.42   
  Averaged      60.53                   61.58              
  Kernel        61.40                   58.77
  ===========   ======================  ======================

We find both version gives similar performances.
