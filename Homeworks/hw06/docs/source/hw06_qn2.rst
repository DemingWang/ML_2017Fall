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

Example of dense.txt file::

    Let's say we have total number of features (num of vocab words) is 4.
    Let's say an examle has sparse svm light file notation:
    1 1:1 2:1 3:1
    
    Then, the first digit 1 is the label.
    1:1 2:1 3:1 is the sparse data.
    
    The 1th, 2nd, and 3rd element is 1 and all other elemtns are zero.
    1:1 2:1 3:1 ==> 1 1 1 0

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

   Vanilla Perceptron Statistics
   =============================
   Final iteration = 18
   Total mistakes = 476


   Parameter w = [ 10.  -1.   3. ...,  -6.  -4.   4.]
   Accuracy    = 97.80 % (978 out of 1000 correct)


   F1-score = 0.97
   Accuracy = 0.98
   Confusion matrix is given below
   Diagonals are True values.
          True_0 True_1
          --------------
   Pred_0| 683      8
   Pred_1| 12      295

Qn 2e Averaged Perceptron
-----------------------------------------
We use the dataset dense_train.txt to train our averaged
perceptron model and get the paramters vector **w**.

Then we fit our model to test data (dense_test.txt) and
found the accuracy.

Result::

  Averaged Perceptron Statistics
  =============================
  Final iteration = 18
  Total mistakes = 476


  Parameter w = [ 9.6  -0.85  2.85 ..., -5.2  -3.45  2.8 ]
  Accuracy    = 98.10 % (981 out of 1000 correct)


  F1-score = 0.97
  Accuracy = 0.98
  Confusion matrix is given below
  Diagonals are True values.
         True_0 True_1
         --------------
  Pred_0| 685      8
  Pred_1| 11      296

Qn 2f Kernel Perceptron (Extra Work)
-----------------------------------------

The statistics for kernel is given below::

  Final iteration = 6
  Total mistakes = 309


    Accuracy    = 98.00 % (980 out of 1000 correct)


    F1-score = 0.97
    Accuracy = 0.98
    Confusion matrix is given below
    Diagonals are True values.
           True_0 True_1
           --------------
    Pred_0| 680      13
    Pred_1| 7      300

Summary of spam-nonSpam Classification
----------------------------------------

Results for spam vs non-spam classification::


  ============  ==========   ==========
  Perceptron    Accuracy     F1-score
  ============  ==========   ==========
  Vanilla       0.98         0.97      
  Averaged      0.98         0.97
  Kernel        0.98         0.97
  ===========   ==========   ==========
