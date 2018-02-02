#!python
# -*- coding: utf-8 -*-#
"""
SVM changing C and gamma

@author: Bhishan Poudel

@date: Nov 19, 2017
https://chrisalbon.com/machine-learning/svc_parameters_using_rbf_kernel.html
"""
# Import packages to visualize the classifer
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings

# Import packages to do the classifying
import numpy as np
from sklearn.svm import SVC
np.random.seed(0) # seed 100 gives bad two regions

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
        
# generate data
def plot_data(X_xor,y_xor):
    plt.scatter(X_xor[y_xor == 1, 0],
                X_xor[y_xor == 1, 1],
                c='b', marker='x',
                label='1')
    plt.scatter(X_xor[y_xor == -1, 0],
                X_xor[y_xor == -1, 1],
                c='r',
                marker='s',
                label='-1')

    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.close()
    
def linear_svc(X_xor,y_xor):
    # Create a SVC classifier using a linear kernel
    svm = SVC(kernel='linear', C=1, random_state=0)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()   

def gamma_0d01(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
def gamma_1(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=1, C=1)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
def gamma_10(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=1)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
def gamma_100(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=100, C=1)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
def C_1(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=0.01, C=1)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
def C_10(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=0.01, C=10)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
def C_1000(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=0.01, C=1000)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
def C_10000(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=0.01, C=10000)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
def C_100009(X_xor,y_xor):
    # Create a SVC classifier using an RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=0.01, C=100000)
    # Train the classifier
    svm.fit(X_xor, y_xor)

    # Visualize the decision boundaries
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()

def main():
    """Run main function."""
    # data
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                           X_xor[:, 1] > 0)    
    y_xor = np.where(y_xor, 1, -1)
    # plot_data(X_xor,y_xor)
    # linear_svc(X_xor, y_xor)
    # gamma_0d01(X_xor,y_xor)
    # gamma_1(X_xor,y_xor)
    # gamma_10(X_xor,y_xor)
    C_100009(X_xor,y_xor)

if __name__ == "__main__":
    main()
