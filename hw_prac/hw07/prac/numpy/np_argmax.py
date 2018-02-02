import numpy as np

x = np.array([ [2,1,4],
               [5,20,7],
               [5,20,70],
               [30,8,18]])

h = np.argmax(x, axis=1)
print("h = {}".format(h))
print("x.shape = {}".format(x.shape))
