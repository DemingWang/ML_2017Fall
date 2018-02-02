Qn 3: Softmax Regression with Minibatch Gradient Descent
========================================================================

In this question we use the softmax regression model with minibatch gradient
descent method to predict the labels of 55,000 MNIST digits samples.

I used the scipy library to implement the minibatch gradient descent model.

To use the gradient descent model, we should first shuffle the data and
then take only some subsamples (batches) to train the model and find the best
fit parameters theta.

The code snippet to shuffle and split the data is like this::

  # shuffle data
  np.random.seed(100)
  perm_idx = np.random.permutation(X.shape[0])
  X2 = X[perm_idx]
  y2 = y[perm_idx]

  # After shuffling split the data
  data2 = np.append(X2,y2,axis=1)
  batches = np.array_split(data2, num_splits)


Here, I took the batch of size 100 examples at a time and I ran the model
2000 epoches to estimate the best fit parameters theta.

To use the gradient descent, we need to tune the hyperparameter `eta` using
the validation set.

In this question there is 55k training data, 10k test data, and there should be
5k validation data drawn randomly from the original 60k train data from the
original MINST data.

However, for this pariticular problem we are not provided the validation data.
So I just chose a reasonable number for eta. I opted eta to be `0.1` and trained
the model. After 9 minutes I got result with about 92.6 % accuracy.

Note that we can tune the parameter eta like in the homework 2, where we chose
the eta that gives the smooth looking decreasing curve for a different values
of eta like eta = [0.001, 0.01, 0.1, 1.0, 10.0] and plot the cost history.

The final results are shown below::

  eta = 10   Accuracy ~ 92.6%    Time ~ 9 mins
  eta = 1.0  Accuracy ~ 92.6%    Time ~ 9 mins
  eta = 0.1  Accuracy ~ 92.220%  Time ~ 9 minutes

The code snippet to perform minibatch gradient descent is given below::

  max_iters, batchsize = 2000, 100
  for n in range(max_iters):
      for i, batch in enumerate(batches):
          img, lbl = batches[i][:,:-1], batches[i][:,-1]
          gradient = softmaxGrad(theta, numClasses, inputSize, decay, img.T, lbl)
          theta = theta - eta/batchsize * gradient
