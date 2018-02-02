import numpy as np


def gen_lin_separable_data(fdata,ftrain,ftest,n_samples):
    """This function will crete three data files. 
    Last 20 examples are taken as test data.
    """
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, size=int(n_samples/2))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, size=int(n_samples/2))
    y2 = np.ones(len(X2)) * -1
    
    
    with open(fdata,'w')  as fo, \
         open(ftrain,'w') as fo1, \
         open(ftest,'w')  as fo2:
        for i in range( len(X1)):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X2[i][0], X2[i][1], y2[i])
            fo.write(line)
            fo.write(line2)
        
        for i in range( len(X1) - 20):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X2[i][0], X2[i][1], y2[i])
            fo1.write(line)
            fo1.write(line2)
        
        for i in range((len(X1) - 20), len(X1) ):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(X2[i][0], X2[i][1], y2[i])
            fo2.write(line)
            fo2.write(line2)
            
def main():
    """Run main function."""
    fdata = '../data/separable/data.txt'
    ftrain = '../data/separable/train.txt'
    ftest = '../data/separable/test.txt'
    n_samples = 100
    gen_lin_separable_data(fdata,ftrain,ftest,n_samples)

if __name__ == "__main__":
    main()
