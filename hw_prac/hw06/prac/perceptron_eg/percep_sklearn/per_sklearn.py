#!python
# -*- coding: utf-8 -*-#
"""
Multi-class Perceptron sklearn

@author: Bhishan Poudel

@date: Nov 14, 2017
http://stamfordresearch.com/scikit-learn-perceptron/
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import perceptron

def perc_sklearn():
    # Data
    d = np.array([
    [2, 1, 2, 5, 7, 2, 3, 6, 1, 2, 5, 4, 6, 5],
    [2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7]
    ])
     
    # Labels
    t = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    
    # plot
    colormap = np.array(['r', 'k'])
    plt.scatter(d[0], d[1], c=colormap[t], s=40)
    # plt.show()
    plt.close()
    
    # rotate the data 180 degrees
    d90 = np.rot90(d)
    d90 = np.rot90(d90)
    d90 = np.rot90(d90)
     
    # Create the model
    net = perceptron.Perceptron(max_iter=100, verbose=0, random_state=100, fit_intercept=True, eta0=0.002)
    net.fit(d90,t)
     
    # Print the results
    print ("Prediction " + str(net.predict(d90)))
    print ("Actual     " + str(t))
    print ("Accuracy   " + str(net.score(d90, t)*100) + "%")


    # Plot the original data
    plt.scatter(d[0], d[1], c=colormap[t], s=40)
     
    # Output the values
    print ("Coefficient 0 " + str(net.coef_[0,0]))
    print ("Coefficient 1 " + str(net.coef_[0,1]))
    print ("Bias " + str(net.intercept_))
     
    # Calc the hyperplane (decision boundary)
    ymin, ymax = plt.ylim()
    w = net.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(ymin, ymax)
    yy = a * xx - (net.intercept_[0]) / w[1]
     
    # Plot the line
    plt.plot(yy,xx, 'k-')
    # plt.show()
    plt.close()

    # test with more data
    
    nX = np.random.random_integers(10, size=(2,50))
     
    # Have a look at it
    # print (nX)
     
    # Rotate it the same as the previous data
    nX90 = np.rot90(nX)
    nX90 = np.rot90(nX90)
    nX90 = np.rot90(nX90)
    
    # Set predication as the predication results
    prediction = net.predict(nX90)
     
    # Print to have a look
    print (prediction)  
    
    # Plot the nX random values AND the prediction as a color
    plt.scatter(nX[0],nX[1], c=colormap[prediction],s=40)
    # Now plot the hyperplane
    ymin, ymax = plt.ylim()
    # Calc
    w = net.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(ymin, ymax)
    yy = a * xx - (net.intercept_[0]) / w[1]
    plt.plot(yy,xx, 'k-')
    plt.show()
    plt.close()

def main():
    """Run main function."""
    perc_sklearn()


if __name__ == "__main__":
    main()
