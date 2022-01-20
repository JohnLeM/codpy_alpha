import os,sys
import numpy as np
import pandas as pd
import time as time
import codpy.codpy as cd
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:sys.path.append(parentdir)
from common_include import *
from data_generators import * 
from predictors import * 
from clusteringCHK import *
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import tensorflow as tf

class crossval_label_codpy_predictor(linear_model.LinearRegression):
    def __init__(self, estimator,**kwargs):
        super().__init__()
        self.estimator = estimator
    def fit(self,X,y,**kwargs):
        out = super().fit(X,y)
        self.param = kwargs
        self.estimator.x = X
        self.estimator.y = X
        self.estimator.fx = y
        self.estimator.z = X
        self.estimator.D, self.estimator.Nx, self.estimator.Ny, self.estimator.Nz = self.estimator.x.shape[1], self.estimator.x.shape[0], self.estimator.y.shape[0], X.shape[0]
        return self
    def predict(self,X,**kwargs):
        super().predict(X,**kwargs)
        self.estimator.z = X
        self.estimator.D, self.estimator.Nx, self.estimator.Ny, self.estimator.Nz = self.estimator.x.shape[1], self.estimator.x.shape[0], self.estimator.y.shape[0], X.shape[0]
        self.estimator.predictor(set_codpy_kernel = set_kernel, **self.param)
        return self.estimator.f_z
    def score(self,*args, **kwargs):
        out = super().score(*args, **kwargs)
        return out
    def get_params(self, deep=True):
        return {"estimator": self.estimator}

set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2,1e-8 ,map_setters.set_standard_min_map)            

iris = datasets.load_iris()
X = iris.data[:150]
y = iris.target[:150]

# print("XGBClassifier")
# pred = XGBClassifier()
# print(cross_val_score(crossval_label_codpy_predictor(estimator = pred), X, y, cv=2))
print("RFClassifier")
pred = RandomForestClassifier()
print(cross_val_score(crossval_label_codpy_predictor(estimator = pred), X, y, cv=2))
X, y = Boston_data_generator().get_X_y_data()
print("ScipyRegressor")
pred = ScipyRegressor()
print(cross_val_score(crossval_label_codpy_predictor(estimator = pred), X, y, cv=2))
tf_param = {'epochs': 50,
'batch_size':16,
'validation_split':0.1,
'loss': tf.keras.losses.mean_squared_error,
'optimizer':tf.keras.optimizers.Adam(0.001),
'activation':['relu','relu','relu','linear'],
'layers':[8,64,64,1],
'metrics':['mse']
}
print("Tensorflow")
pred = tfRegressor()
print(cross_val_score(crossval_label_codpy_predictor(estimator = pred), X, y, cv=2, fit_params = tf_param))

