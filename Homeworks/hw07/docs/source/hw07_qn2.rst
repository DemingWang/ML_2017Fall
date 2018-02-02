Qn2: Text Classification
============================
In this question, we trained SVM model to classify spam vs non-spam and atheism vs
religion problem. We choose linear kernel and cost parameter 5. ::

  from sklearn.svm import SVC
  model = SVC(kernel='linear', random_state=0, C=5.0)
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  correct = np.sum(y_test == y_pred)
  accuracy = correct / len(y_test)

For the SVM method the results is following::
    
    spam: accuracy = 0.977
    Confusion Matrix
    [[683  10]
     [ 13 294]]

Comparison with Perceptron method is given below::
    
      Results for spam vs non-spam classification::

      ======================  ==========
      Model                    Accuracy
      ======================  ==========
      Vanilla  Perceptron     0.98
      Averaged Perceptron     0.98
      Kernel   Perceptron     0.98
      Linear   SVM            0.977
      =====================   ==========

Newgroups Classification
---------------------------
For SVM model, the results for newsgrops classification is given below::
    
    news1: accuracy = 0.6649122807017543
    Confusion Matrix
    [[149 102]
     [ 89 230]]

    news2: accuracy = 0.6298245614035087
    Confusion Matrix
    [[136 115]
     [ 96 223]]

Comparison of SVM with Perceptron model::
 
      # Results for 20 newsgroups atheism vs religion classification::

      ==============    ======================  ======================
      Model             Version 1 Accuracy %    Version 2 Accuracy %
      ==============    ======================  ======================
      Vanilla  Perc     59.12                   58.42
      Averaged Perc     60.53                   61.58
      Kernel   Perc     61.40                   58.77
      Linear SVM        66.49                   62.98
      =============     ======================  ======================

Conclusion
-----------
I got better result for 20 newsgroup classification and almost same result
for spam classification.
