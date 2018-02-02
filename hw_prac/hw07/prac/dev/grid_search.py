from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

import numpy as np
from sklearn import preprocessing
np.set_printoptions(3)

def min_max_scale(fdata):
    """ 
    X_norm = X - X_min / (X_max - X_min)
    
    Min is calculated for a column by default.
    """
    *X,t = np.genfromtxt(fdata,unpack=True,dtype='f',delimiter=',')
    X = np.array(X).T
    
    ## minmax scalar
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    X_norm = scaler.fit_transform(X.T).T   # each rows
    
    return X_norm, t

def grid_search():  
    fdata = '../data/optdigits/train/train.txt'
    # fdata = '../data/optdigits/devel/devel.txt'
    X,y = min_max_scale(fdata)
    

    # model = SVC(kernel='linear', random_state=0, C=1.0, cache_size=4000, class_weight='balanced')
    # params = np.arange(1,21)
    # grid = GridSearchCV(model, dict(max_iter=params))
    # grid.fit(X,y)
    # best_max_iter = grid.best_estimator_.max_iter
    # print("best_max_iter = {}".format(best_max_iter))

    model = SVC(kernel='poly', random_state=0, C=1.0, cache_size=4000, class_weight='balanced',max_iter=11)
    params = { "degree": [2,3,4,5,6]}
    grid = GridSearchCV(model, params)
    grid.fit(X,y)
    best_degree = grid.best_estimator_.degree
    print("best_degree = {}".format(best_degree))
    # print("grid.best_estimator_.max_iter = {}".format(grid.best_estimator_.max_iter)) # -1
    
    # model = SVC(kernel='rbf', random_state=0, C=1.0, cache_size=4000, class_weight='balanced',max_iter=11)
    # gammas = 0.5 * np.array([0.1,0.5,2,5,10]) ** -2 # gamma = 1/2/sigma**2
    # grid = GridSearchCV(model, dict(gamma=gammas))
    # grid.fit(X,y)
    # best_gamma = grid.best_estimator_.gamma
    # print("best_gamma = {}".format(best_gamma)) # 0.125
    # print("best_sigma = {}".format(np.sqrt(1/2/best_gamma))) # 2.0
    
    
def main():
    """Run main function."""
    grid_search()

if __name__ == "__main__":
    main()
