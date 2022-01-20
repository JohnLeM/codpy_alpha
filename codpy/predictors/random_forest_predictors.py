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


class RandomForestRegressor(data_predictor, add_confusion_matrix):

    def copy(self):
        return self.copy_data(RandomForestRegressor())

    def get_model(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        

        return RandomForestRegressor(n_estimators=n_estimators, max_depth= max_depth).fit(self.x,self.fx)

    def predictor(self,**kwargs):
        model = self.get_model()
        self.f_z = model.predict(self.z)

    def id(self,name = "Rforest"):
        return "RForest"

class RandomForestClassifier(codpyprClassifier, add_confusion_matrix):

    def copy(self):
        return self.copy_data(RandomForestClassifier())

    def predictor(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        n_estimators = kwargs.get('n_estimators',400)
        num_jobs = kwargs.get('num_jobs', -1)

        RandomFor = RandomForestClassifier(n_estimators = n_estimators)
        RandomFor.fit(self.x, self.fx)
        self.f_z = RandomFor.predict(self.z)


    def id(self,name = "Rforest"):
        return "RForest"


    


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

    RF_param = {'max_depth': 5,
        'n_estimators': 5}
    
    scenarios_list = [ (1, 100*i, 50,100*i ) for i in np.arange(1,5,1)]
    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    RandomForestRegressor(set_kernel = set_per_kernel),
    data_accumulator(), **RF_param)
    results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
    print(results)
    list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 2)


