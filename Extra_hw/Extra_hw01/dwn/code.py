#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error



# Read data matrix X and labels t from text file.
def read_data(file_name,M):
#  YOUR CODE here:
  dataset = np.loadtxt(file_name)
  X = dataset[:,:-1]
  # Making the vander matrix
  X = np.vander(X[:,0],M + 1, increasing =True)
  t = dataset[:,dataset.shape[1] - 1]
  print("t.shape = ", t.shape)

  return X, t




# Implement normal equations to compute w = [w0, w1, ..., w_M].
def train(X,t):
#  YOUR CODE here:
  w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)


  return w


def train_penaltize(X,t,lambda_opt, M):
  I = np.eye(M + 1)
  w_pen = np.linalg.inv(X[:,0:M + 1].T.dot(X[:,0:M+1]) +
                        np.exp(lambda_opt) * len(t) * I ).dot(X[:,:M+1].T).dot(t)

  return w_pen


# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
#  YOUR CODE here:
  E_RMS = np.sqrt(((np.dot(X,w.T) - t) ** 2).mean())
  E_RMS = mean_squared_error(X@w, t)**0.5
  return E_RMS


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
#  YOUR CODE here:
  J_w = np.sum((np.dot(X,w.T) - t) ** 2)/len(t)/2
  return J_w




##======================= Main program =======================##
parser = argparse.ArgumentParser('Multivariate Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/polyfit',
                    help='Directory for the multivariate houses dataset.')
FLAGS, unparsed = parser.parse_known_args()


# Read the training, test and validation data sets with appropriate
# degree of polynamial which here is 0 to 9.
M = 9
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt",M)
Xtest, ttest   = read_data(FLAGS.input_data_dir + "/test.txt", M)
Xdevel, tdevel = read_data(FLAGS.input_data_dir + "/devel.txt",M)


# Now let's calculate E_rms for train and test sets
E_train = []
E_test  = []





for i in range(0,M+1):
  w = train(Xtrain[:,0:i+1], ttrain)
  E = compute_rmse(Xtrain[:,0:i+1], ttrain, w)
  # print('loop ', i, ': rmse = ', float(E))
  # print("ttrain[0] = ", ttrain[0])
  E_train = np.append(E_train, compute_rmse(Xtrain[:,0:i+1], ttrain, w))
  E_test  = np.append(E_test, compute_rmse(Xtest[:,0:i+1] , ttest, w))


# Plotting
M_poly = np.arange(10)
plt.plot(M_poly, E_train, color = "blue", label = "train", marker = "o")
plt.plot(M_poly, E_test,  color = "red" , label = "test" , marker = "*")
plt.xlabel("M")
plt.ylabel("$E_{rms}$")
plt.legend()
plt.title("unregularized")
plt.savefig("poly_test_train.png")
plt.show()
plt.close()




# Computing the E_rms for the regularized.
lambda_opt = np.arange(-50, 5, 5)




E_train_opt = []
E_devel_opt  = []
for j in range(0,len(lambda_opt)):
  w_pen = train_penaltize(Xtrain,ttrain,float(lambda_opt[j]), M)
  E_train_opt = np.append(E_train_opt, compute_rmse(Xtrain,
                                                    ttrain, w_pen))
  E_devel_opt = np.append(E_devel_opt, compute_rmse(Xdevel ,
                                                    tdevel, w_pen))
print("##########################################################")
print("Regularization method:")
print("The minimum E_rms for the validation data %2f:"% min(E_devel_opt))
minimum = np.where(E_devel_opt == min(E_devel_opt))
print("The lambda that makes E_rms minimum is %2f" % lambda_opt[minimum])




plt.plot(lambda_opt, E_train_opt, color = "red", marker = "o", label = "train")
plt.plot(lambda_opt, E_devel_opt, color = "blue", marker = "o", label = "test")
plt.xlabel("ln $\lambda$")
plt.ylabel("$E_{rms}$")
plt.legend()
plt.title("poly_regularied")
plt.savefig("poly_regularied.png")
plt.show()
plt.close()




print("")
print("###################################################")
print("Here we compare the E_rms between regularized and non-regulrized methods.")
w = train(Xtrain, ttrain)
E_rms_test = compute_rmse(Xtest, ttest, w)
print('Test RMSE without regulariztion for M = 9: %0.4f.' % E_rms_test)


w_pen = train_penaltize(Xtrain,ttrain,float(lambda_opt[minimum]), M)
E_rms_test_opt = compute_rmse(Xtest, ttest, w_pen)
print('Test RMSE with regulariztion for M = 9: %0.4f.' % E_rms_test_opt)
