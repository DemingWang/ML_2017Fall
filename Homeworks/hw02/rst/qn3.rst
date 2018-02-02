Multivariate Linear Regression
==================================

In this question we fit batch gradient descent (BGD)  method to find the
house prices of given multivariate data.


a) The plot of J(w) versus the number of epochs with step size 10 is shown
below:

.. image:: ../../images/multi_cost_history.png

b) After training I printed the parameters (w) for train data and compared
with normal equations. For bgd I used 500 iterations and learning rate of 0.1.
I found that weight vectors are exact same.::

    Train Data
  epochs  lr     mean       std       w0          w1          rmse        cost
  500     0.1    2340.46     808.44    254,450.00  93,308.92   64,083.51   2,053,348,364.32

  w_norm_eqn = [[254,450.0000  78,097.1126  24,424.5992  2,079.7085 ]]
  w_bgd      = [[254,449.9998  78,079.1811  24,442.5758  2,075.9564 ]]


c) Comparison of BGD and SGD.

In stochastic grad desc, I shuffled the data while training. To get the
reproducible results, I set the random seed to 100. The SGD training gives
noisy results. The weight vector does not stabilize smoothly, it first decreases
rapidly then fluctuates with some noise but after some iterations it will give
similar results like BGD.

For this example after 468 epochs I got similar weight vectors from bgd and sgd::

  np.random.seed(100)
  w_norm_eqn        = [[254,450.0000  78,097.1126  24,424.5992  2,079.7085 ]]
  iters = 500 w_BGD = [[254,450.0000  78,097.1122  24,424.5995  2,079.7084 ]]
  iters = 499 w_SGD = [[254,516.1325  78,063.9833  24,487.8213  2,003.9658 ]]
  abs_diff_min = 238.22658943321608
  len(abs_diff_min_lst) = 500
  np.argmin(abs_diff_min_lst) =  468
