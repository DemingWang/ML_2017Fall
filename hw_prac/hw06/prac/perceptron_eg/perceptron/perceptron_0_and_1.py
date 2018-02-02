import numpy as np

def perceptronLearn(D, maxIter=10, alpha=0.1, b=0):
    w = np.zeros(np.shape(D)[1] - 1)
    iterCounter = 0
    while iterCounter < maxIter:
        iterCounter += 1
        print(("Iteration " + str(iterCounter) + " " + ("-" * 20)))
        iterError = 0
        for row in D:
            x = row[:-1]
            d = row[-1]
            y = 1 if w.dot(x) + b > 0 else 0
            print(("    wanted: " + str(d) + ", got: " + str(y)))
            w += alpha * (d - y) * x
            iterError += np.mean(np.abs(d - y))
        if iterError == 0:
            break
    print(("Returning weights: " + str(w)))
    return w

'''
Training Matrix.
Each row vector is composed of feature vector entries followed by the 
desired output value.
This particular matrix features the logical NOT-AND operation as an example.
'''
D = np.array([
          [1, 0, 0, 1],
          [1, 0, 1, 1],
          [1, 1, 0, 1],
          [1, 1, 1, 0],
    ])

perceptronLearn(D)
