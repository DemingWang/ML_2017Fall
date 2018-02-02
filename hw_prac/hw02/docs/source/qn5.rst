Stochastic Gradient Descent
============================

Here I have implemented SGD to questions 1, 2, and 3 using the same hyperparameters.
All the comparisons are given in the respective questions.

SGD converged faster than GD mehtod, however, the final weight vectors
oscillates around the values that we obtained from normal equations method.


For univariate case::

  np.random.seed(100)
  w_norm_eqn        = [[254,450.0000  93,308.9201 ]]
  iters = 200 w_BGD = [[254,449.9998  93,308.9200 ]]
  iters = 199 w_SGD = [[254,494.4429  93,244.2965 ]]
  abs_diff_min = 109.06649000954349
  np.argmin(abs_diff_min) =  175



For multivariate case::

  np.random.seed(100)
  w_norm_eqn        = [[254,450.0000  78,097.1126  24,424.5992  2,079.7085 ]]
  iters = 500 w_BGD = [[254,450.0000  78,097.1122  24,424.5995  2,079.7084 ]]
  iters = 499 w_SGD = [[254,516.1325  78,063.9833  24,487.8213  2,003.9658 ]]
  abs_diff_min = 238.22658943321608
  len(abs_diff_min_lst) = 500
  np.argmin(abs_diff_min_lst) =  468

For polynomial case:

For unregularized case::

  # NOTE: I used threshold = 0.001 not 1e-10 for Unregularized cased (bgd and sgd)
  np.random.seed(100) # for SGD to get same results
  shrinkage = 0.00 iters = 238 learning_rate = 0.10 deg = 5 threshold = 1.00e-04
  w_norm_eqn  = [[0.1711  1.6086  5.8353  -35.1236  43.5586  -16.1171 ]]
  w_unreg_bgd = [[0.1711  -0.0314  -0.6621  -0.3889  0.1045  0.5930 ]]
  w_unreg_sgd = [[0.1747  -0.0231  -0.6722  -0.3940  0.1107  0.6114 ]]


For regularized case::

  np.random.seed(100) # for SGD to get same results
  shrinkage  = 0.10 final_iter = 657 learning_rate = 0.10 deg = 5 threshold = 1.00e-10
  w_norm_eqn = [[0.1711  -0.1535  -0.3607  -0.1988  0.0442  0.2763 ]]
  w_reg_bgd  = [[0.1555  -0.1537  -0.3606  -0.1987  0.0442  0.2762 ]]
  w_reg_sgd  = [[0.0890  -0.1079  -0.1054  -0.0680  -0.0268  0.0098 ]]
