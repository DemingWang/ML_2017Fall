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

3.2d Use the tuned hyperparameters T,d,sigma on Training data to test the accuracy of test data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we have the tuned parameters::
  
  tuned_T = 4
  tuned_d = 6
  tuned_sigma = 10
  
Using these values we tested the one-vs-rest multiclass perceptron on the testing data.
The summary of results is given below::
  
  Model                           Accuracy(%)    # of Sup. Vecs  Runtime
  ------                          -----------   --------------  --------      
  Vanilla Perceptron              92.487        N/A             2 seconds 
  Linear Kernel Perceptron        91.04         317             1 min 39 seconds
  Polynomial Kernel Perceptron    95.492        157             1 min 58 seconds    
  Gaussian Kernel Perceptron      29.1597       491             9 min 14 seconds

  Polynomial Kernel Perceptron achieves best performance.
  Gaussian Kernel Perceptron takes longest time to run. 
  Since it has to  compute exponentials of all the elements and 
  have to create a large Gram matrix.
  
  NOTE: for polynomial kernel the true digit 1 was most confused with predicted 4.


The outputs for vanilla perceptron is given below::

  Accuracy = 92.48747913188647

  y_true is y-axis
  y_pred is x_axis
  
  [[177   0   0   0   0   1   0   0   0   0]
   [  0 143  15   0   5   1   1   0   5  12]
   [  0   0 175   0   1   0   0   1   0   0]
   [  1   0   7 155   0   3   0   2   1  14]
   [  0   0   0   0 177   0   0   1   3   0]
   [  0   0   1   0   0 176   0   1   0   4]
   [  1   0   0   0   2   0 178   0   0   0]
   [  0   0   0   0   1   9   0 161   1   7]
   [  1  10   2   1   1   5   1   0 149   4]
   [  0   1   0   0   3   2   0   0   3 171]]

  Begin time:  Fri Dec  1 21:50:23 2017
  End   time:  Fri Dec  1 21:50:24 2017 

  Time taken:  0 days,  0 hours,        0 minutes,  1.652384 seconds.

The outputs for Linear Kernel Perceptron is given below::

  Accuracy = 91.04062326099054
  Number of support vectors = 317


  [[176   0   0   0   0   2   0   0   0   0]
   [  0 137  16   0   9   2   5   0   2  11]
   [  0   0 175   0   1   0   0   1   0   0]
   [  1   0   9 156   0   3   1   3   0  10]
   [  0   0   0   0 177   0   0   1   1   2]
   [  0   0   1   1   0 178   0   1   0   1]
   [  1   0   0   0   2   0 178   0   0   0]
   [  0   0   0   0   1   9   0 164   1   4]
   [  4   8   6   2   4   7  10   0 127   6]
   [  1   1   1   1   5   2   0   0   1 168]]

  Begin time:  Fri Dec  1 21:53:57 2017
  End   time:  Fri Dec  1 21:55:35 2017 

  Time taken:  0 days,  0 hours,        1 minutes,  38.330967 seconds.


The outputs for Polynomial Kernel Perceptron is given below::

  Accuracy = 95.49248747913188
  Number of support vectors = 157


  [[178   0   0   0   0   0   0   0   0   0]
   [  0 182   0   0   0   0   0   0   0   0]
   [  0   6 168   0   0   0   0   1   2   0]
   [  0   1   1 173   0   1   0   1   3   3]
   [  0  20   0   0 161   0   0   0   0   0]
   [  0   2   0   0   0 176   2   0   0   2]
   [  2   2   0   0   0   0 177   0   0   0]
   [  0   1   0   0   0   2   0 164   1  11]
   [  0  12   0   0   0   0   0   0 161   1]
   [  0   1   0   0   0   1   0   0   2 176]]

  Begin time:  Fri Dec  1 22:00:01 2017
  End   time:  Fri Dec  1 22:01:59 2017 

  Time taken:  0 days,  0 hours,        1 minutes,  57.998752 seconds.


The outputs for Gaussian Kernel Perceptron is given below::

  Accuracy = 29.15971062882582
  Number of support vectors = 491


  [[  0  64 113   0   0   0   0   1   0   0]
   [  0 182   0   0   0   0   0   0   0   0]
   [  0  52 125   0   0   0   0   0   0   0]
   [  0 165  15   0   0   0   0   3   0   0]
   [  0  77   0   0 104   0   0   0   0   0]
   [  0 153  13   0   0   0   0  16   0   0]
   [  0 162  15   0   0   0   0   4   0   0]
   [  0  66   0   0   0   0   0 113   0   0]
   [  0 172   1   0   0   0   0   1   0   0]
   [  0 151  19   0   0   0   0  10   0   0]]

  Begin time:  Fri Dec  1 22:03:27 2017
  End   time:  Fri Dec  1 22:12:41 2017 

  Time taken:  0 days,  0 hours,        9 minutes,  13.341146 seconds.
