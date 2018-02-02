from sklearn import datasets
import numpy as np
 

def perceptron(X, Y,epochs):
    """
    X: data matrix without bias.
    Y: target
    """
    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X1 = np.append(ones, X, axis=1)
      
    w = np.zeros(X1.shape[1])
    final_iter = epochs
    
    for epoch in range(epochs):        
        misclassified = 0
        for i, x in enumerate(X1):
            y = Y[i]
            h = np.dot(x, w) * y

            if h <= 0:
                w = w + x*y
                misclassified += 1

        if misclassified == 0:
            final_iter = epoch
            break
        
    print("final_iter = {}".format(final_iter))                
    return w, final_iter

def main():
    """Run main function."""
    iris = datasets.load_iris() 
    XX = iris.data[0:100, :2]  # we only take the first two features. y = iris.target
    y = iris.target[0:100]
    
    w,final_iter = perceptron(XX, y,20000)
    print("w = {}".format(w))
    print("final_iter = {}".format(final_iter))

if __name__ == "__main__":
    main()
