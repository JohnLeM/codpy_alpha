import os, sys
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.append(parentdir)
data_path = os.path.dirname(__file__)
from common_include import * 
from data_generators import *



class ScipyRegressor(data_predictor):
    def copy(self):
        return self.copy_data(ScipyRegressor())

    
    def predictor(self,**kwargs):
        from scipy.interpolate import Rbf
        if (self.D*self.Nx*self.Ny*self.Nz ):
            tx = list(np.transpose(self.x))
            tz = list(np.transpose(self.z))
            tx.append(self.fx)
            rbf = Rbf(*tx,**kwargs)
            self.f_z = rbf(*tz)
    def id(self,name = ""):
        return "scipy pred"


def rbf_test_regressor(scenarios_list,**kwargs): 

    set_per_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)

    def my_fun(x):
        import numpy as np
        from math import pi
        D = len(x)
        res = 1.;
        for d in range(0,D):
            res *= np.cos(4 * x[d] * pi) 
        for d in range(0,D):
            res += x[d]
        return res

    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    ScipyRegressor(set_kernel = set_per_kernel),
    data_accumulator(), **kwargs)
    results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1)
    print(results)


if __name__ == "__main__":
    import tensorflow as tf
    scenarios_list = [ (1, 100, 100, 100) ]
    kwargs = {'rescale:xmax': 1000,
    'rescale:seed':42,
    'sharp_discrepancy:xmax':1000,
    'sharp_discrepancy:seed':30,
    'sharp_discrepancy:itermax':10,
    'discrepancy:xmax':500,
    'discrepancy:ymax':500,
    'discrepancy:zmax':500,
    'discrepancy:nmax':2000}

    rbf_param = {'function': 'gaussian',
    'epsilon':None,
    'smooth':0.,
    'norm':'euclidean'}

    rbf_test_regressor(scenarios_list,**{**kwargs,**rbf_param})        