Qn 2: Scipy Implementation of Softmax Regression
=====================================================

In this question we implement the softmax regression with scipy library.
We evaluate the digit recognition task from the mnist data.

1 C Parameter
---------------
The cost function with L2 regularizaion in scikit library is given below:

When the target labels t_n belongs to {-1,1 } then:
.. math::
    
  J(w) = C \sum_{n=1}^N ln C ( 1 + e^{-t_n W^T x_n}) + \frac{1}{2} W^T W
  
But when the target label t_n belongs to {0,1} then the L2 regularized cost 
function for logistic regression is given by:

.. math::
    
  J(w) = \sum_{n=1}^N ln C ( 1 + e^{-t_n W^T x_n}) + \frac{\lambda}{2} W^T W
  
While getting the maximum likelihood estimate of parameter w, we have:

.. math::
    
  \hat{w} = \operatorname*{argmin}_w J(w)
  
And we get :math:`C = \frac{1}{\lambda}`.

In our case lambda or decay parameter is 1e-4, so C parameter is 1e+4.

2 Softmax  training
----------------------
To train our logistic regression model I used sklearn package 
`linear_model.LogisticRegression` with `lbfgs` solver.

To get the reproducible result I used the argument `random_state=100`.
I choose only 100 iterations and multinomial for softmax regression.


To make the code efficient I used all the cores of the computer by using 
argument n_jobs=4. To find how many cores do we have in our computer, we 
can use the multiprocessing library and the command cpu_count.

The code snippet is given below::
    
    import multiprocessing
    n_jobs =  multiprocessing.cpu_count()
    print("n_jobs = {}".format(n_jobs)) # 4
    softmax = linear_model.LogisticRegression(C=C, penalty='l2', random_state=100,
                  solver='lbfgs', max_iter=100, multi_class='multinomial',n_jobs=n_jobs, verbose=1)

    softmax.fit(X_train, y_train)

3 Softmax testing
---------------------
After training the logistic regression model using training examples, I tested
this model with test examples. I got 92.6 % accuracy on the testing dataset.

The code snippet is given below::
    
    # predict
    y_pred = softmax.predict(X_test)
    acc = np.mean(y_test == y_pred)
    print('Accuracy: %0.3f%%.' % (acc * 100)) # 92.670%.
