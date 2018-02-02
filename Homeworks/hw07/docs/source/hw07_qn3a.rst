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
  
Qn 3a: Data Processing (One-vs-Rest break data into c classes and normalize)
------------------------------------------------------------------------------
Perceptron is a binary classifier, by default it can not classify multi-class
dataset. To use perceptron for multi-class classification we need to create
binary datasets from the original dataset. Here, we use one-vs-rest method
to create the binary dataset.

We want to use the optical digits data from the UCI website:
http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

The database has two data files, training data, and test data:

  - optdigits.tra # 3833 training examples
  - optdigits.tes # 1797 test examples


We take all the test examples as test examples. But, from the training examples
we reserve some examples for the validation set. We choose first 1000 examples as
the development data and rest 2833 examples as the training data.

Now the dataset looks like this:

  - train.txt # 2833 examples
  - devel.txt # 1000 examples
  - test.txt # 1797 examples

Description of data::
    
    Each example is a 8*8 grayscale image of a digit (0-9).
    The pixel value ranges from 0 to 16, and are separated by comma.
    
    An example has 8 * 8 = 64 features and one label.
    First 64 values are features and last value is label.
    
    For instance, an example may look like this:

    0,0,0,6,16,2,2,0,0,0,3, ..., 4 

    First 64 values are features and last value is the label.

Feature Scaling (min-max scaling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In Perceptron algorithm (or any algorithms in general) we get better accuracy
if we rescaled our training examles. Here we adopt min-max scaling to scale each
examples. We scale all the examples, that means training data, devel data and 
test data.

Use one-vs-rest method create binary data from multiclass data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each example has 64 features and last value a number between 0-9.
We first rescale first 64 features using min-max scaling so that each values
are in between 0 to 1.

Then we need to make the last value either 1 or -1. We use the one-vs-rest
method and create 10 dataset from the single dataset. Here we have 10 classes, 
so there are 10 datafiles for each of the dataset: traning, devel, and test.
