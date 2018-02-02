import numpy as np


# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)

  return mean, std


# Standardize the features of the examples in X by subtracting their mean and
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):
  S = (X - mean) / std

  return S

def checking(X):
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler = scaler.fit(X)
        print('Mean: {}, StandardDeviation: {}'.format (scaler.mean_, np.sqrt(scaler.var_)))
        normalized = scaler.transform(X)
        print("normalized[0] = {}".format(normalized[0]))

    except:
        print('ERROR: the library sklearn not found!')


def main():
    """Run main function."""
    *X,t = np.genfromtxt('../data/multivariate/train.txt',unpack=True,dtype=np.float64)
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

    # checking using sklearn
    checking(X)


if __name__ == "__main__":
    main()
