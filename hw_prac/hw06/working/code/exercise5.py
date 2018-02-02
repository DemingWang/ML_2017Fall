from perceptron import perceptron_train
from perceptron import aperceptron_train
from perceptron import perceptron_test
from perceptron import quadratic_kernel
from perceptron import kperceptron_train
from perceptron import kperceptron_test
import numpy as np
np.set_printoptions(2)

def quadratic_kernel(x, y):
    return ( 1 + np.dot(y,x))**2
    
def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:,:-1]
    Y = data[:,-1]
    
    return X, Y

def run_perceptron(X, Y, epochs,fout):
    
    w, final_iter,mistakes = perceptron_train(X, Y, epochs,verbose=True)
    score = perceptron_test(w, X)

    correct  = np.sum(score == Y)
    accuracy = correct/ len(score) * 100
    
    with open(fout,'w') as fo:
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        print("Perceptron Accuracy = {:.2f} % ({} out of {} correct)".format(
               accuracy, correct, len(score)),file=fo)

def run_aperceptron(X, Y, epochs,fout):
    
    w, final_iter,mistakes = aperceptron_train(X, Y, epochs,verbose=True)
    score = perceptron_test(w, X)

    correct  = np.sum(score == Y)
    accuracy = correct/ len(score) * 100
    
    with open(fout,'a') as fo:
        print("\n",file=fo)
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        print("Averaged Perceptron Accuracy = {:.2f} % ({} out of {} correct)".format(
               accuracy, correct, len(score)),file=fo)

def run_kperceptron(X, Y, epochs,fout):
    alpha, sv, sv_y,final_iter,mistakes = kperceptron_train(X,Y,epochs,quadratic_kernel,verbose=1) 
    score = kperceptron_test(X,quadratic_kernel,alpha,sv,sv_y,minus_one=False)
    
    correct = np.sum(score == Y)
    accuracy = correct/ len(score) * 100
    
    with open('outputs/outputs.txt','a') as fo:
        print("\n",file=fo)
        print("Final iteration = {}".format(final_iter), file=fo)
        print("Total mistakes = {}".format(mistakes), file=fo)
        print("alpha = {}".format(alpha),file=fo)
        print("Kernel Perceptron Accuracy = {:.2f} % ({} out of {} correct)".format(
              accuracy, correct, len(score)),file=fo)

def main():
    """Run main function."""

    X, Y = read_data('../data/ex5/ex5.txt')
    epochs = 20
    fout = 'outputs/outputs.txt'

    # vanill, averaged, kernel perceptron
    run_perceptron(X, Y, epochs,fout)
    run_aperceptron(X, Y, epochs,fout)
    run_kperceptron(X, Y, epochs,fout)



if __name__ == "__main__":
    main()
