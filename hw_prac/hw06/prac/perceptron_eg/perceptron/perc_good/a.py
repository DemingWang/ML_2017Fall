# Imports
import numpy as np
import random
import os, subprocess
np.set_printoptions(2)



def generate_points( N):
    xA,yA,xB,yB = [random.uniform(-1, 1) for i in range(4)]
    V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])

    X = []
    for i in range(N):
        x1,x2 = [random.uniform(-1, 1) for i in range(2)]
        x = np.array([1,x1,x2])
        s = int(np.sign(V.T.dot(x)))
        X.append((x, s))
    return X


def main():
    """Run main function."""
    X = generate_points(5)
    print(X)

if __name__ == "__main__":
    main()
