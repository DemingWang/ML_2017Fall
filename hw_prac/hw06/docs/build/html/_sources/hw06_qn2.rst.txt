Qn 2: Spam vs. Non-Spam
=====================================================

In this question we will implement two perceptron algorithms
  - perceptron
	- averaged perceptron

Here, we have two datasets:
  - spam_train.txt (4000 examples lines)
  - spam_test.txt  (1000 examples lines)

For example: the data.txt looks like this
.. code::
  
  1 there is an apple and a banana
  0 a cow in a farm a pig in a farm
  1 an apple in the tree


Qn 2a Create feature vector (vocab.txt)
-----------------------------------------

The first entry in the `data.txt` is the label (spam or not-spam). We create feature matrix (vocab.txt) such that 
any word in data.txt should appear in at least TWO examples.

Then vocab.txt looks like this:
.. code::
  
  1 a
  2 an
  3 apple
  4 in
  
NOTE: In actual homework we choose least word frequecy to be 30.

In actual homework::
  
  function: spam_exercise.py==>create_vocab(fdata,min_freq,fvocab)
  
  datafile: ../data/spam/spam_train.txt
  vocabfile: ../data/spam/spam_vocab.txt


Qn 2b Create sparse feature matrix
-----------------------------------------

Here, we create sparse feature matrix `sparse.txt` from
the feature matrix vocab.txt.

The example of sparse.txt is given below::
  
  1 1:1 2:1 3:1
  0 1:1 4:1
  1 2:1 3:1 4:1
  
  Words common between data_line1 & vocab:  'a', 'an','apple' 
  Their positions in vocab.txt : 1 2 3
  All values after colon are 1.

In actual homework::
  
  function: spam_exercise.py==>create_sparse(fdata,fvocab,fsparse)
  
  vocab file: ../data/spam/spam_vocab.txt
  sparse file train: ../data/spam/spam_train_svm.txt
  sparse file test: ../data/spam/spam_test_svm.txt
  
Qn 2c Read dense matrix
-----------------------------------------

Here, first we create the dense feature matrix (dense.txt)
and take this as the input design matrix for the perceptron
algorithm. The first column of the design matrix is label
and rest part of the matrix is the data.


In actual homework::
  
  function: spam_exercise.py==>create_sparse(fdata,fvocab,fsparse)
  
  function: perceptron.py==>read_examples(file_name)
  
  dense file train: ../data/spam/dense_train.txt
  dense file test: ../data/spam/dense_test.txt
  
  labels : first column of dense.txt
  data   : all columns except first column of dense.txt

Qn 2d Vanilla Perceptron
-----------------------------------------
Here we read the data and labels from dense.txt file and 
implement the perceptron algorithm to train the data.
We find following things:

  - number of epochs to converge train data  
  - total number of mistakes
  - final parameter vector w
  
  We write these things into a text file spam_model_p.txt.
  
 Then we test our perceptron model, we first get the parameter
 vector **w** from training data (dense_train.txt). 
 Then we train our model
 to the test dataset (dense_test.txt).

 Result::
   
   Test accuracy = %.
  
Qn 2e Averaged Perceptron
-----------------------------------------
We use the dataset dense_train.txt to train our averaged
perceptron model and get the paramters vector **w**.

Then we fit our model to test data (dense_test.txt) and 
find the accuracy.
