import numpy as np

# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  mean = np.mean(X, axis=0,keepdims=True)
  std = np.std(X, axis=0, keepdims=True)

  return mean, std


# Standardize the features of the examples in X by subtracting their mean and
# dividing by their standard deviation, as provided in the parameters.
# Here we find z-scores.
def standardize(X, mean, std):
    """Normalize the given array.

    Here we use Z-score normalization

    :math:`z = \\frac{x - \mu}{\sigma}`

    There is also another normalization method called Min-Max Scaling
    :math:`X_{norm} = \\frac{X - X_{min}}{X_{max} - X_{min}}`

    """
    S = (X - mean) / std

    return S

def checking():
    from scipy import stats
    *X,t = np.genfromtxt('../data/multivariate/train.txt',unpack=True,dtype=None)
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)
    print("\n\nscipy.stats checking")
    print("X.shape = {}".format(X.shape))
    print("t.shape = {}".format(t.shape))

    S = stats.zscore(X, axis=0)
    print("S[0] = {}".format(S[0]))




def main():
    """Run main function."""
    *X,t = np.genfromtxt('../data/multivariate/train.txt',unpack=True,dtype=None)
    X,t,t = np.array(X).T, np.array(t), t.reshape(len(t),1)
    print("X.shape = {}".format(X.shape))
    print("t.shape = {}".format(t.shape))
    # print(X)
    # print("t = ", t)
    print("X.mean(axis=0) = {}".format(X.mean(axis=0)))

    # checking
    mean, std = mean_std(X)
    print("mean = {}".format(mean))
    print("std = {}".format(std))

    # checking S
    S = standardize(X, mean, std)
    print("S[0] = {}".format(S[0]))

    # checking using scipy.stats
    checking()


if __name__ == "__main__":
    main()
