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


class AdaBoostRegressor(data_predictor, add_confusion_matrix):

    def copy(self):
        return self.copy_data(AdaBoostRegressor())

    def get_model(self, **kwargs):
        from sklearn.ensemble import AdaBoostRegressor
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        base_model = kwargs.get('base_model', 'tree_model')
        tree_no = kwargs.get('tree_no', 50)
        learning_rate = kwargs.get('learning_rate', 1)

        return AdaBoostRegressor(n_estimators=tree_no, learning_rate = learning_rate).fit(self.x,self.fx)

    def predictor(self,**kwargs):
        model = self.get_model()
        self.f_z = model.predict(self.z)

    def id(self,name = "AdaBoost"):
        return "AdaBoost"

class AdaBoostClassifier(codpyprClassifier):

    def copy(self):
        return self.copy_data(AdaBoostClassifier())
    
    def get_model(self, **kwargs):
        from sklearn.ensemble import AdaBoostClassifier
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        base_model = kwargs.get('base_model', 'tree_model')
        tree_no = kwargs.get('tree_no', 50)
        learning_rate = kwargs.get('learning_rate', 0.1)
        algorithm = kwargs.get('algorithm', 'SAMME.R')
        
        return AdaBoostClassifier( n_estimators=tree_no, learning_rate= learning_rate,
         algorithm=algorithm, random_state=0).fit(self.x,self.fx)
    
    def predictor(self, **kwargs):
        model = self.get_model()
        self.f_z = model.predict(self.z)


    def id(self,name = "AdaBoost"):
        return "AdaBoost"


    


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

    ada_param = {'tree_no': 50,
    'learning_rate': 1}
    
    scenarios_list = [ (1, 100*i, 50,100*i ) for i in np.arange(1,5,1)]
    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    AdaBoostRegressor(set_kernel = set_per_kernel),
    data_accumulator(), **ada_param)
    results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
    print(results)
    list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 2)


