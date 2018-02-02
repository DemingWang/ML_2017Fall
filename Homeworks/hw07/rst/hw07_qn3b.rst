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
  x0          x0w0, x0w1,..., x0w9         x0w3       h0 = 3        y0 = 8
  x1          x1w0, x1w1,..., x1w9         x1w5       h1 = 5        y1 = 5
  ...
  xn          xnw0, xnw1,..., xnw9         x0w7       hn = 7        yn = 7
  
  Accuracy = 90 % for epochs 1.
  Accuracy = 88 % for epochs 2.
  Accuracy = maximum % for epoch tuned_epoch.
  
  I found tuned_epoch = 13
  File: code/outputs/tune_T.txt
  Then we will tune another paramters for kernel perceptrons.
  
Results: tuned_epoch = 13
         runtime = 23 seconds

3.2b Tune hyperparameter d for Polynomial Kernel from Development data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I found the tuned value of d = 6 (outputs/tune_d.txt).
Runtime is 9 minutes 54 seconds.


3.2c Tune hyperparameter sigma(or gamma) for Gaussin Kernel from Development data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I found the tuned value of sigma = 10 (outputs/tune_sigma.txt).
Runtime is 41 minutes.

Tuning summary::

  Tuning                             Tuned value   Max Accuracy(%)   Runtime
  -------------------------------   -----------   --------------     ---------    
  epochs (from vanilla perceptron)  13            95.10              23 seconds         
  degree d (for polynomial kernel)  5             96.80              10 minutes     
  sigma (for gaussian kernel)       10            33.00              41 minutes           
