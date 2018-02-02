import numpy as np

def computeNumericalGradient(J, theta):
    """ Compute numgrad = computeNumericalGradient(J, theta)

    Args:
    
        theta: a vector of parameters
        J: a function that outputs a real-number and the gradient.
    
    Return:
        
        numgrad : 1d array of numerical gradient of given cost function.
        numgrad[i] = del J/ del_theta[i]
        
    .. math:: \\frac{d}{d\\theta} = frac{J(\\theta + \\epsilon) - J(\\theta - \\epsilon)}{2 \\epsilon}
    
    Where, epsilon = 0.0001.
    """

    # Initialize numgrad with zeros
    numgrad = np.zeros(theta.size)

    # perturbation
    perturb = np.zeros( theta.size )
    e = 0.0001

    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad
