Qn 4: . spam-nonSpam vs atheism-religion
========================================================================

For the perceptron binary classification of spam vs non-spam we got following
results::

  ============  ==========   ==========
  Perceptron    Accuracy     F1-score
  ============  ==========   ==========
  Vanilla       0.98         0.97
  Averaged      0.98         0.97
  Kernel        0.98         0.97
  ===========   ==========   ==========

For the case of atheism vs religion, we got the following results::

  ============  ======================  ======================
  Perceptron    Version 1 Accuracy %    Version 2 Accuracy %
  ============  ======================  ======================
  Vanilla       59.12                   58.42
  Averaged      60.53                   61.58
  Kernel        61.40                   58.77
  ===========   ======================  ======================

Analysis
----------
In case of spam versus non-spam binay classification we got about 98% accuracy,
however, for atheism vs religion we got around 60% accuracy.

Some of the reasons may be following:

  - Spam classification has larger train-test examples.
    The spam-nonSpam problem has 4,000 training examples and 1,000 test examples,
    however, the 20newsgroups religion vs atheism problem has 857 training 
    examples and 570 test examples. Larger the training dataset, better is the 
    performance of the classifier.

  - The spam and non-spam documents are separated more clearly. There are many words
    that we can think of (and machine will anaylyze) that only belong to
    either spam or non-spam. The document lenght may also vary significantly.
    This means there are more feature vectors we
    can think of that are vital in classifying whether a document is spam or 
    not-spam.

    However, if we look at atheism and religion documents, they both look similar.
    The documents have similar size (total number of words), and many of the words
    may appear in both the documents. The articles about both atheism and religion
    may talk about the similar thing and only the implied meaning is different.
    In terms of machines, there may not be many DISTINCT FEATURES that can
    separate a document from athiesm to relgion if we train a binary classier to
    differentiate them. 
    
    Hence, we can argue that spam vs non-spam is relatively easy problem 
    compared to religion vs atheism classification and machines also reflects
    this idea and gives less accuracy for spam-nonSpam than religion-atheism 
    classification.
