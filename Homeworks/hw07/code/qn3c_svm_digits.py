from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
np.set_printoptions(3)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Save all the Print Statements in a Log file.
# import sys
# old_stdout = sys.stdout
# log_file = open("summary.log","w")
# sys.stdout = log_file


def min_max_scale(fdata):
    """ 
    X_norm = X - X_min / (X_max - X_min)
    
    Min is calculated for a column by default.
    """
    *X,y = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X = np.array(X).T
    
    ## minmax scalar
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    X_norm = scaler.fit_transform(X.T).T   # each rows
    
    return X_norm, y

def grid_search(X_train,y_train,X_devel,y_devel):  
    # model = SVC(kernel='linear', random_state=0, C=1.0)
    # params = np.arange(1,21)
    # grid = GridSearchCV(model, dict(max_iter=params))
    # grid.fit(X,y)
    # best_max_iter = grid.best_estimator_.max_iter
    # print("best_max_iter = {}".format(best_max_iter))

    # model = SVC(kernel='poly', random_state=0, C=1.0)
    # params = { "degree": [2,3,4,5,6]}
    # grid = GridSearchCV(model, params)
    # grid.fit(X,y)
    # best_degree = grid.best_estimator_.degree
    # print("best_degree = {}".format(best_degree))
    # # print("grid.best_estimator_.max_iter = {}".format(grid.best_estimator_.max_iter)) # -1
    
    model = SVC(kernel='rbf', random_state=0, C=1.0, gamma=0.125)
    # model = SVC(kernel='rbf', random_state=0, C=1.0,max_iter=11, gamma=0.125)
    model.fit(X_train,y_train)
       
    y_pred = model.predict(X_devel)
    correct = np.sum(y_devel == y_pred)
    accuracy = correct / len(y_devel)
    print("y_dpred[0:4] = {}".format(y_pred[0:4]))
    print("accuracy = {}".format(accuracy))
    cm = confusion_matrix(y_devel,y_pred)
    cr = classification_report(y_devel,y_pred)
    # print(cm)
    # print(cr)
    
    return y_pred

def plot_digits(test_img,test_labels,test_labels_pred):
    # Show the Test Images with Original and Predicted Labels
    # In image processing we use 255 = white 0 = black 
    # in 256-color grayscale colormap
    #
    # plot some random 10 images
    a = np.random.randint(1,400,10)
    for i in a:
    	two_d = (np.reshape(test_img[i], (8, 8)) * 255).astype(np.uint8)
    	plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[i],test_labels_pred[i]))
    	plt.imshow(two_d, interpolation='nearest',cmap='autumn')
    	plt.show()

def main():
    """Run main function."""
    fdata_train = '../data/optdigits/train/train.txt'
    fdata_devel = '../data/optdigits/devel/devel.txt'
    X_train,y_train = min_max_scale(fdata_train)
    X_devel,y_devel = min_max_scale(fdata_devel)
    
    y_pred = grid_search(X_train,y_train,X_devel,y_devel)
    
    plot_digits(X_devel,y_devel,y_pred)

if __name__ == "__main__":
    main()
