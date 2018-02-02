import numpy as np
from scipy.sparse import coo_matrix

def softmaxCost(theta, numClasses, inputSize, decay, data, labels):
    """ Computes and returns the (cost, gradient)

    Args:
    
        theta: 1d array of parameter.
               example of theta ::
                 numClasses = 10
                 inputSize  = 28 * 28 = 784
                 theta = 0.005 * randn(numClasses * inputSize)
                       = 1d array of shape (7840,)
                     
        numClasses: the number of classes (e.g. 10)
        
        inputSize: the size N of the input vector (e.g. 28 * 28 = 784)
        
        decay : weight decay parameter (e.g. lamda = 1e-4)
        
        data: the N x M input matrix
              where each row data[i, :] corresponds to a single sample
              e.g. 55000, 784
             
        labels: an N x 1 matrix containing the labels corresponding for the input data
                e.g. shape is  (55000,)
             
    Return:

        cost: Cost of the model

        thetagrad : 1d array of gradient values
            
    
    
    .. math::

        E(w) = - \\frac{1}{N} \\sum_{n=1}^N \sum_{k=1}^K \\sumN_delta_k(t_n) ln \\frac{exp(w_k x_n)}{Z(x_n)} + \\
        \\frac{\lambda}{2} \sum_{k=1}^K w_k^2

    Also the gradient is given by:

    .. math:: \\nabla_{w_k} E(w) = - \\frac{1}{N} \\sum_{n=1}^N ( \\sumN_delta_k(t_n) - p(C_k | x_n)) x_n + \\lambda w_k
       
    """

    # Unroll the parameters from theta
    theta = theta.reshape(numClasses, inputSize) # shape (10,784)

    N = data.shape[1] # 55k examples.

    groundTruth = coo_matrix( (  np.ones(N, dtype = np.uint8),
                                 (labels, np.arange(N))
                              )
                             ).toarray()
    cost = 0
    thetagrad = np.zeros((numClasses, inputSize))

    # shapes
    # print("\n\ntheta.shape = {}".format(theta.shape))         # (10, 784)
    # print("data.shape = {}".format(data.shape))               # (784, 55000)
    # print("labels.shape = {}".format(labels.shape))           # (55000,) 
    # print("groundTruth.shape = {}".format(groundTruth.shape)) # (10, 55000)
    
    X = data
    z = theta @ X # shape (10,784) (784,55000) = (10, 55000)
    N = data.shape[1]
    sumN_delta = groundTruth
   
    hyp = np.exp(z-np.amax(z, axis=0, keepdims=True)) # (10, 55000)
    prob = hyp / np.sum(hyp, axis = 0) # (10, 55000)
    cost = np.multiply(sumN_delta, np.log(prob)) # (10, 55000)
    cost = -1/N * np.sum(cost) # (scalar)
    
    weight_decay = 1/2 * decay * np.sum(theta**2)
    cost = cost + weight_decay
    
    # now find gradient of cost function
    # thetagrad shape = (10, 55000) (55000,784) = (10, 784)
    thetagrad = -1/N *  (sumN_delta - prob) @ X.T + decay * theta

    # Unroll the gradient matrices into a vector for the optimization function.
    grad = thetagrad.ravel()

    return cost, grad

def softmaxGrad(theta, numClasses, inputSize, decay, data, labels):
    """ Computes and returns the (cost, gradient)

    Args:
    
        theta: 1d array of parameter.
               example of theta ::
                 numClasses = 10
                 inputSize  = 28 * 28
                 theta = 0.005 * randn(numClasses * inputSize)
                     
        numClasses: the number of classes (e.g. 10)
        
        inputSize: the size N of the input vector (e.g. 28 * 28 = 784)
        
        decay : weight decay parameter (e.g. lamda = 1e-4)
        
        data: the N x M input matrix
              where each row data[i, :] corresponds to a single sample
              e.g. 55000, 784
             
        labels: an M x 1 matrix containing the labels corresponding for the input data
                e.g. shape is  (55000,)
             
    Return:

        cost: Cost of the model

        thetagrad : 1d array of gradient values
            
    
    
    .. math::

        E(w) = - \\frac{1}{N} \\sum_{n=1}^N \sum_{k=1}^K \\sumN_delta_k(t_n) ln \\frac{exp(w_k x_n)}{Z(x_n)} + \\
        \\frac{\lambda}{2} \sum_{k=1}^K w_k^2

    Also the gradient is given by:

    .. math:: \\nabla_{w_k} E(w) = - \\frac{1}{N} \\sum_{n=1}^N ( \\sumN_delta_k(t_n) - p(C_k | x_n)) x_n + \\lambda w_k
       
    """

    # Unroll the parameters from theta
    theta = np.reshape(theta, (numClasses, inputSize))

    numCases = data.shape[1]
    

    groundTruth = coo_matrix((np.ones(numCases, dtype = np.uint8),
                            (labels, np.arange(numCases)))).toarray()
    cost = 0
    thetagrad = np.zeros((numClasses, inputSize))

    # shapes
    # print("\n\n")
    # print("theta.shape       = {}".format(theta.shape))       # (10, 784)
    # print("thetagrad.shape   = {}".format(thetagrad.shape))   # (10, 784)
    # print("data.shape        = {}".format(data.shape))        # (784, 55000)
    # print("data.shape[1]     = {}".format(data.shape[1]))     # 55000
    # print("labels.shape      = {}".format(labels.shape))      # (55000,) 
    # print("groundTruth.shape = {}".format(groundTruth.shape)) # (10, 55000)
    
    X = data
    z = theta @ X # shape (10,784) (784,55000) = (10, 55000)
    N = data.shape[1]
    sumN_delta = groundTruth
   
    hyp = np.exp(z-np.amax(z, axis=0, keepdims=True)) # (10, 55000)
    prob = hyp / np.sum(hyp, axis = 0) # (10, 55000)
    
    # now find gradient of cost function
    # thetagrad shape = (10, 55000) (55000,784) = (10, 784)
    thetagrad = -1/N *  (sumN_delta - prob) @ X.T + decay * theta

    # Unroll the gradient matrices into a vector for the optimization function.
    grad = thetagrad.ravel()

    return  grad

def softmaxPredict(theta, data):
    """Computes and returns the softmax predictions in the input data.

    Args:
    
      theta : model parameters matrix of shape  (numClasses x inputSize) 
              e.g. 10, 784  (Here theta is taken after fitting the model.)
                
      data: the M x N input matrix (transpose of data has N examples.)
        
    Return:
    
      pred : 1d array of predictions where the output values z = theta_X has maximum label value.
             pred.shape = (test_size,) e.g. (10k,) which is same as test_labels shape.
               
    """

    # z is theta_X matrix
    z = data.T @ theta.T
    pred = np.argmax(z, axis=1)
    # print("\n\n")
    # print("theta.shape = {}".format(theta.shape)) # (10, 784)
    # print("data.shape = {}".format(data.shape))   # (784, 10000)
    # print("pred.shape = {}".format(pred.shape))   # (10000, ) (same as test_labels)
    # print("\n\n")

    # ------------------------------------------------------------------
    return pred
