Qn 3: Digit Recognition
=========================
In this problem learn to classify the optical digits using Perceptron and SVM:

  - Perceptron
  
    + Vanilla Perceptron
    + Linear Kernel Perceptron
    + Polynomial Kernel Perceptron (degree= 2,3,4,5,6)
    + Gaussian Kernel Perceptron (sigma = 0.1, 0.5, 2, 5, 10)
    
  - Support Vector Machines (SVM)
  
    + Linear SVM
    + Polynomial Kernel SVM
    + Gaussian Kernel SVM
  
Qn 3.1 One vs. Rest Multiclass Classification
---------------------------------------------------
For the one-vs-rest multiclass classification using perceptron,
we first download the data from UCI website:
http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

We download two files:

  - optdigits.tra
  - optdigits.tes

From the original training data, we chose first 1000 examples as
the development data and rest 2833 examples as the training data.  
In case of test data, we took all the 1797 test examples as test data.

All the examples contain 65 numbers separated by comma.
For example the first example of training data looks like this::

  0,0,0,6,15,2,2,0,0,0,3, ..., 4 
  # each example is a gray-scale image of grid 8*8 = 64 features
  # last digit is the label.
  # Here, the last digit 4 is the label.
  # Initial 64 digits are the features.
  
  # bash command to look first example
  head -1 train.txt
  tail -1 train.txt

3.1a MaxMin Scale the Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For each examples we used max-min scaling method to scale the 
examples between 0 and 1 (from 0-16,inclusive). The scaled files are train_norm.txt, devel_norm.txt, and test_norm.txt.

3.1b break data into 10 parts (one-vs-rest)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
  It will run until the heat death of the universe, or until the earth will be
  consumed by a black hole.
  
  Let's say for epoch T =1, we have following case:
  
  examples    10_predictions               maximum    hypothesis   True label
  --------    ---------------------        -------    ----------   -----------
  x0          x0w0, x0w1,..., x0w9         x0w3       h0 = 3    y0 = 8
  x1          x1w0, x1w1,..., x1w9         x1w5       h1 = 5    y1 = 5
  ...
  xn          xnw0, xnw1,..., xnw9         x0w7       hn = 7    yn = 7
  
  Accuracy = 90 % for epochs 1.
  Accuracy = 88 % for epochs 2.
  Accuracy = maximum % for epoch tuned_epoch.
  
  I found tuned_epoch = 4.
  File: code/outputs/tune_epochs.txt
  Then we will tune another paramters for kernel perceptrons.

3.2b Tune hyperparameter d for Polynomial Kernel from Development data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I found the tuned value of d = 6 (outputs/tune_d.txt).


3.2c Tune hyperparameter sigma(or gamma) for Gaussin Kernel from Development data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I found the tuned value of sigma = 6 (outputs/tune_sigma.txt).

3.2d Use the tuned hyperparameters T,d,sigma on Training data and test with test data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we have the tuned parameters::
  
  tuned_T = 4
  tuned_d = 6
  tuned_sigma = 0.5
  
Using these values we tested the one-vs-rest multiclass perceptron on the testing data.
The summary of results is given below::
  
  Model                           Accuracy    # of Sup. Vecs  Runtime
  ------                          ---------   --------------  ----      
  Vanilla Perceptron
  Linear Kernel Perceptron
  Polynomial Kernel Perceptron
  Gaussian Kernel Perceptron

  Gausssian Kernel Perceptron achieves best performance.
  Gaussian Kernel Perceptron takes longest time to run. 
  Since it has to  compute exponentials of all the elements and 
  have to create a large Gram matrix.


The confusion matrix for vanilla perceptron is given below:

The confusion matrix for Linear Kernel Perceptron is given below:

The confusion matrix for Polynomial Kernel Perceptron is given below:

The confusion matrix for Gaussian Kernel Perceptron is given below:
