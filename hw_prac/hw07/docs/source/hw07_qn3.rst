Qn 3: Digit Recognition
=========================
In this problem we write following alogrithm to classify the optical digits:

  - Perceptron without kernel
  - Perceptron with kernel
  - SVM without Kernel
  - SVM with  Kernel
  
Qn 3.1
---------
 
First we download the data from UCI website:
http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

We download two files:

  - optdigits.tra
  - optdigits.tes
  
We split the training data into development and training data.
First 1000 examples are taken as development data and rest 2833 examples are taken as training data. All the 1797 test examples are taken as test data.

All the examples contain 65 words separated by comma::

  awk -F '[, \t]+' '{print NF}' train.txt
  # gives 65 8*8 = 64 is the feature size and last digit is label
  
  # look first example
  head -1 train.txt
  tail -1 train.txt

3.1a normalize the features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For each examples we used max-min scaling method to normalize the 
examples between 0 and 1 (from 0-16,inclusive). The normalized files are train_norm.txt, devel_norm.txt, and test_norm.txt.

3.1b break data into 10 parts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each of the dataset (train,devel,test) example has last digit label from 0-9.
We create 10 copies of the same dataset such that one of the label is
set to ONE and rest of the labels are set to MINUS-ONE.

For example, from train_norm.txt we create train0.txt, ..., train9.txt,
from devel_norm.txt we create devel0.txt to devel9.txt and so on.

Qn 3.2 Linear, Polynomial and Gaussian Kernel Perceptron
------------------------------------------------------------
In this problem we train different perceptron models to our digits dataset.

3.2a Tune hyperparameter T (number of epochs) for Vanilla Perceptron from Development data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We use training to train the linear perceptron model for 
different epochs T = 1,2,...,20.

Example::
 
  Before we put out feet on the training data, we go through the development
  phase. We use our development data to tune the number of epoch hyperparameter.
  
  We do not run perceptron until convergence to train our development dataset. 
  If the dataset is not linearly separable, perceptron will never converge. 
  It will run until the heat death of the universe, or the earth will be
  consumed by a black hole.
  
  Let's say for epoch T =1, we have following case:
  
  examples    10_predictions               maximum    prediction   True label
  --------    ---------------------        -------    ----------   -----------
  x0          x0w0, x0w1,..., x0w9         x0w3       h0 = x0w3    y0 = 8
  x1          x1w0, x1w1,..., x1w9         x1w5       h1 = x1w5    y1 = 5
  ...
  xn          xnw0, xnw1,..., xnw9         x0w7       hn = x0w7    yn = 7
  
  For 20 epochs we have 20 different accuracies and we choose the epochs that 
  gives the maximum accuracy.
  We take this value of epoch for all the rest of this homework.
  After we get the best epoch T* we run the linear perceptron with this number
  of epochs.
  
  Then we will tune another paramters for kernel perceptrons.

3.2b Tune hyperparameter d for Polynomial Kernel from Development data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


3.2c Tune hyperparameter sigma(or gamma) for Gaussin Kernel from Development data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3.2d Used tuned hyperparameters T,d,sigma on Training data and test with test data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
