Qn 4: . spam-nonSpam vs atheism-religion
========================================================================
For the perceptron binary classification of spam vs non-spam we got following
results:

============  ======================
Perceptron    Accuracy %    
============  ======================
Vanilla       98.00
Averaged      98.10
Kernel        98.00
===========   ======================

For the case of atheism vs religion, we got the following results:

============  ======================  ====================== 
Perceptron    Version 1 Accuracy %    Version 2 Accuracy %
============  ======================  ====================== 
Vanilla       60.53                   61.23   
Averaged      60.70                   61.75               
Kernel        61.40                   61.05
===========   ======================  ======================

Analysis
----------
In case of spam versus non-spam binay classification we got about 98% accuracy,
however, for atheism vs religion we got around 61% accuracy.

It is because classifying spam vs non-spam is relatively easy job. There are
certain words that appear in spam and not appear in non-spam. There is almost 
universal agreement between which is spam and which is not.

However, in case of classifying whether a document is religious or atheist,
there is not a universal agreement. Also, there are so many religions in the
world and it makes the classification difficult.
