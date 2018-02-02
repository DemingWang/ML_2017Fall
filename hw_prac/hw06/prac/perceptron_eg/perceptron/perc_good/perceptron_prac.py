#!python
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

X = np.array([[2,1],[3,4],[4,2],[3,1]])
# Y = np.array([0,0,1,1]) #orig
Y = np.array([-1,-1,1,1])
h = .02  # step size in the mesh


# we create an instance of SVM and fit our data. We do not scale our
# data since we want to plot the support vectors

clf = Perceptron(n_iter=100).fit(X, Y)

# print(X)
idx0 = np.where(np.array(Y)==0)
idx0 = np.where(np.array(Y)==-1)
idx1 = np.where(np.array(Y)==1)
X0 = X[idx0]
X1 = X[idx1]

fig, ax = plt.subplots()
ax.plot(X0[:,0],X0[:,1],'r_', label="Negative class")
ax.plot(X1[:,0],X1[:,1],'g+', marker='+', label="Positive class")
# ax.scatter(X0[:,0],X0[:,1],c='r', marker='_', label="Negative class")
# ax.scatter(X1[:,0],X1[:,1],c='b', marker='+', label="Positive class")


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))



# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].

w = clf.coef_   # w = [[ 7. -9.]]
w = np.array(w.ravel()[::-1])  # [-9.  7.]

# print('w = ', w)
# a = clf.predict([1,3])
# print('a = ', a)

def predict(X,w):
    return np.sign(np.dot(X, w.T))

# print("X.shape = {}".format(X.shape))
print("w.shape = {}".format(w.shape))
print("xx.ravel().shape = {}".format(xx.ravel().shape))

xxr = xx.ravel().reshape(len(xx.ravel()), 1)
yyr = yy.ravel().reshape(len(yy.ravel()), 1)
print("xxr.shape = {}".format(xxr.shape))

v=np.append(xxr,yyr,axis=1)
print("\n")
print("v.shape = {}".format(v.shape))
print("v[0] = {}".format(v[0]))

z = predict(v,w)
z = z.ravel()
print("z.shape = {}".format(z.shape))
print("z[0] = {}".format(z[0]))


vals = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(vals)
# Z = clf.predict(vals).reshape(50000,1)
print("\n")
print("vals.shape = {}".format(vals.shape)) # 50k, 2
print("Z.shape = {}".format(Z.shape)) # 50k,
print("z.shape = {}".format(z.shape)) # 50,

# compare
print("\n")
print('w = ', w) # w =  [[ 7. -9.]]
print("xx.ravel()[0], yy.ravel()[0] = {} {}".format(xx.ravel()[0], yy.ravel()[0])) # 1.0 0.0
print("Z[0] = {}".format(Z[0])) # 0  negative
print("z[0] = {}".format(z[0])) # 1 positive

# w = -9,7
# x,y = 1.28,0.0  Z=-1  z=-1
# x,y = 1.3,.0;0 Z=1    z=-1
# 14 th point: 1.28 0.0 -1 = -1.0 w: [-9.  7.]
# 15 th point: 1.3 0.0 1 = -1.0 w: [-9.  7.]
# whatever I choose w or w[::-1] I can not get this!!
# I don't know how sklearn predict works!!!

# 
# for i,_ in enumerate(Z):
#     print(i, "th point:", xx.ravel()[i], yy.ravel()[i],  Z[i], "=", z[i], "w:", w)
#     if Z[i] != z[i]:
#         print("Not equal")
#         print(i, "th point:", xx.ravel()[i], yy.ravel()[i], Z[i], "=", z[i], "w:", w)
#         break

# Put the result into a color plot
Z = Z.reshape(xx.shape)
z = z.reshape(xx.shape)



# print("\n")
# print("Z.shape = {}".format(Z.shape)) # 50k,
# print("z.shape = {}".format(z.shape))

ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
# ax.contourf(xx, yy, z, cmap=plt.cm.Paired)
# ax.axis('off')

# Plot also the training points
# ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

ax.set_title('Perceptron')
# plt.show()
