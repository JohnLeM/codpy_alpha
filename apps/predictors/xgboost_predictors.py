import os, sys
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.append(parentdir)
data_path = os.path.dirname(__file__)
from common_include import * 
from data_generators import *
from codpy_predictors import *
import torch.nn as nn
import torch


class XGBRegressor(data_predictor):

    def get_params(**kwargs): return kwargs.get('XGBRegressor',{})
    def get_model(self, **kwargs):
        import xgboost as xgb 
        import os
        xgb_param = XGBClassifier.get_params(**kwargs)
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        n_estimators = xgb_param.get('n_estimators', 10)
        max_depth = xgb_param.get('max_depth', 5)
        return xgb.XGBRegressor(booster = 'gbtree', n_estimators=n_estimators, max_depth= max_depth).fit(self.x,self.fx)

    def predictor(self,**kwargs):
        model = self.get_model()
        self.f_z = model.predict(self.z)

    def id(self,name = "XGboost"):
        return "XGboost"

class XGBClassifier(codpyprClassifier):

    def get_params(**kwargs): return kwargs.get('XGBClassifier',{})


    def get_model(self, **kwargs):
        import xgboost as xgb 
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        xgb_param = XGBClassifier.get_params(**kwargs)

        self.train = xgb.DMatrix(self.x, label=self.fx)
        self.test = xgb.DMatrix(self.z)

        return xgb.train(xgb_param, self.train)

    def predictor(self,**kwargs):
        model = self.get_model(**kwargs) 
        self.f_z = model.predict(self.test)

    def id(self,name = "XGboost"):
        return "XGboost "


if __name__ == "__main__":
    set_per_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)
    def my_fun(x):
        import numpy as np
        from math import pi
        D = len(x)
        res = 1.
        for d in range(0,D):
            res *= np.cos(4 * x[d] * pi) 
        for d in range(0,D):
            res += x[d]
        return res

    xgb_param = {'max_depth': 5,
        'n_estimators': 10}
    
    scenarios_list = [ (1, 100*i, 50,100*i ) for i in np.arange(1,5,1)]
    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    XGBRegressor(set_kernel = set_per_kernel),
    data_accumulator(), **xgb_param)
    results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
    print(results)
    list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 2)


