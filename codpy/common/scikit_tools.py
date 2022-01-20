import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

def RBF_m(x, fx, z, function = "gaussian",norm = "euclidean"):
    from scipy.interpolate import Rbf
    import numpy as np
    # use RBF method
    tx = list(np.transpose(x))
    tz = list(np.transpose(z))
    tx.append(fx)
    rbf = Rbf(*tx)
    fz = rbf(*tz)
    return fz

def fun_logreg_accuracy(**kwargs):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter = 500000)
    model.fit(kwargs['x'], kwargs['fx'])
    accuracy = model.score(kwargs['z'], kwargs['fz'])
    return accuracy

def fun_svm_accuracy(**kwargs):
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    classifier = SVC(kernel = 'rbf', random_state = 5)
    classifier.fit(kwargs['x'], kwargs['fx'])
    accuracies = cross_val_score(estimator = classifier, X = kwargs['z'], y = kwargs['fz'], cv = 5)
    return accuracies.mean()

def cross_validation_score(x):
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    #indices for time series
    splits = TimeSeriesSplit(n_splits=5)
    plt.figure(1)
    index = 1
    for train_index, test_index in splits.split(x):
        train = alg.sampler(x[train_index], np.shape(x[train_index])[0])
        test = alg.sampler(x[test_index], np.shape(x[test_index])[0])
        print('Observations: %d' % (np.shape(train)[0] + np.shape(test)[0]))
        print('Training Observations: %d' % (np.shape(train)[0]))
        print('Testing Observations: %d' % (np.shape(test)[0]))
        plt.subplot(510 + index)
        plt.plot(train)
        plt.plot([None for i in train] + [j for j in test])
        index += 1
        print(r2_score(x[test_index], test))
    plt.show()  
