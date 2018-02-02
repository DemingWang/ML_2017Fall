import numpy as np
np.set_printoptions(2)

sigma = np.array([0.1, 0.5, 2, 5, 10])

gamma = 0.5 * sigma**-2

print("gamma = {}".format(gamma))
print("1/2/0.5**2 = {}".format(1/2/0.5**2))
