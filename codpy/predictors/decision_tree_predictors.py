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


class DecisionTreeRegressor(data_predictor):

    def copy(self):
        return self.copy_data(DecisionTreeRegressor())

    def get_model(self, **kwargs):
        from sklearn.tree import DecisionTreeRegressor
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        max_depth = kwargs.get('max_depth', 10)
        
        return DecisionTreeRegressor(max_depth= max_depth).fit(self.x,self.fx)

    def predictor(self,**kwargs):
        model = self.get_model()
        self.f_z = model.predict(self.z)

    def id(self,name = "Decision tree"):
        return "Decision tree"

class DecisionTreeClassifier(codpyprClassifier, add_confusion_matrix):

    def copy(self):
        return self.copy_data(DecisionTreeClassifier())
    
    def get_model(self, **kwargs):
        from sklearn.tree import DecisionTreeClassifier
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        criterion = kwargs.get('criterion', 'gini')
        splitter = kwargs.get('splitter', 'random')
    
        return DecisionTreeClassifier(criterion='gini',splitter= splitter)
    
    def predictor(self, **kwargs):
        model = self.get_model().fit(self.x, self.fx)
        self.f_z = model.predict(self.z)

    def get_proba(self, scenarios, **kwargs):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import label_binarize

        x = scenarios.accumulator.get_xs()[0]
        z = scenarios.accumulator.get_zs()[0]
        fx = scenarios.accumulator.get_fxs()[0]
        classifier = OneVsRestClassifier(self.get_model())
        classes = np.unique(fx)
        y = label_binarize(fx, classes = classes)
        return classifier.fit(x, y).predict_proba(z)

    def id(self,name = "Decision tree"):
        return "Decision tree"


    


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

    DT_param = {'max_depth': 10}
    
    scenarios_list = [ (1, 100*i, 50,100*i ) for i in np.arange(1,5,1)]
    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    DecisionTreeRegressor(set_kernel = set_per_kernel),
    data_accumulator(), **DT_param)
    results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
    print(results)
    list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 2)


