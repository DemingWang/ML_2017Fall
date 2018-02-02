import numpy as np

def computeNumericalGradient(J, theta):
    """ Compute numgrad = computeNumericalGradient(J, theta)

    Args:
    
        theta: a vector of parameters
        J: a function that outputs a real-number and the gradient. (cost,grad)
    
    Return:
        
        numgrad : 1d array of numerical gradient of given cost function.
        numgrad[i] = del J/ del_theta[i]
        
    .. math:: \\frac{d}{d\\theta} = frac{J(\\theta + \\epsilon) - J(\\theta - \\epsilon)}{2 \\epsilon}
    
    Where, epsilon = 0.0001.
    
    Usage:
    
    
      This function has two args J and theta and gives numGrad output.
      In the file softmaxExercise.py we use this function::
      
          if FLAGS.debug:
              checkNumericalGradient()

              # note: cost, grad = softmaxCost(theta, numClasses, inputSize, decay, images, labels)
              numGrad = computeNumericalGradient(
                          lambda x: softmaxCost(x, numClasses, inputSize, decay, images, labels),
                          theta
                          )

              # Use this to visually compare the gradients side by side.
              print(np.stack((numGrad, grad)).T)


              # Compare numerically computed gradients with those computed analytically.
              diff = norm(numGrad - grad) / norm(numGrad + grad)
              print(diff)
              sys.exit(1)
      
      
    """

    # Initialize numgrad with zeros
    # numgrad has same shape as theta, its just derivative of theta.
    # np.size = np.prod(a.shape) = product of dims = total num of elements.
    numgrad = np.zeros(theta.size)

    # perturbation
    perturb = np.zeros(theta.size)
    e = 0.0001

    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        cost1, _ = J(theta - perturb)  # func J gives two outputs cost,grad
        cost2, _ = J(theta + perturb)
        
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (cost2 - cost1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad
