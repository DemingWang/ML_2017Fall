import numpy as np
np.set_printoptions(2)
import matplotlib.pyplot as plt

# Print the hyperplane calculated by svm_sgd()
w = [1,2]
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = list(zip(*x2x3))

print('w = ', w)
print("x2 = {}".format(x2))
print("x3 = {}".format(x3))
print("X = {}".format(X))
print("Y = {}".format(Y))
print("U = {}".format(U))
print("V = {}".format(V))

ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')
ax.quiver(1,2,3,4,color='r',scale=1)

plt.grid(True)
plt.show()
plt.close()
