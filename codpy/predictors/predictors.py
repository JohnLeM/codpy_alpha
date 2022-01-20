import os, sys
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.append(parentdir)
data_path = os.path.dirname(__file__)
from common_include import * 
from data_generators import *
from codpy_predictors import *
from scikit_predictors import *
from scipy_predictors import *
from tensorflow_predictors import *
from pytorch_predictors import *
from xgboost_predictors import *
from random_forest_predictors import *
from decision_tree_predictors import *
from svm_predictors import *
from adaboost_predictors import *
from gradientboosting_predictors import *




####A standard run

def standard_supervised_run(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs):
    scenario_generator.run_scenarios(scenarios_list,generator,predictor,accumulator,**kwargs)
    if bool(kwargs.get("Show_results",True)):
        results = accumulator.get_output_datas().dropna(axis=1)
        print(results)
    if bool(kwargs.get("Show_confusion",False)):accumulator.plot_confusion_matrices(**kwargs,mp_title = "confusion matrices for "+predictor.id())
    if bool(kwargs.get("Show_maps",False)):print(accumulator.get_maps_cluster_indices())
    list_results = [(s.z,s.f_z) for s in scenario_generator.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 6)


####predictors


##################################### Kernels
set_gaussian_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8,map_setters.set_mean_distance_map)
set_tensornorm_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 0,0,map_setters.set_unitcube_map)
set_per_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)
##################################### Blob
    
def random(**codpy_param):
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

    set_kernel = set_per_kernel
    scenarios_list = [ (1, i, i,i ) for i in np.arange(100,300,100)]
    validator_compute=['accuracy_score','discrepancy_error','norm_function']
    scenario_generator_ = scenario_generator()
    
    standard_supervised_run(scenario_generator_,scenarios_list,data_random_generator(fun = my_fun),standard_Tensorflow_predictor(set_kernel = set_per_kernel),data_accumulator(),validator_compute = validator_compute, type = "sto",**codpy_param)
    standard_supervised_run(scenario_generator_,scenarios_list,data_random_generator(fun = my_fun),standard_scipy_predictor(set_kernel = set_per_kernel),data_accumulator(),validator_compute = validator_compute,type = "sto", **codpy_param)
    standard_supervised_run(scenario_generator_,scenarios_list,data_random_generator(fun = my_fun),standard_codpy_predictor(set_kernel = set_per_kernel),data_accumulator(),validator_compute = validator_compute,type = "sto",**codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Nx","scores"),("Nx","discrepancy_errors"),("Nx","norm_function"),("Nx","execution_time")],
        mp_title = "Benchmark methods"
    )
    # scenario_generator_.compare_plot(axis_label = "Ny",field_label="discrepancy_errors")
    # ###################################

def main_test(**codpy_param):
    random(**codpy_param)


if __name__ == "__main__":
    kwargs = {'rescale:xmax': 1000,
    'rescale:seed':42,
    'sharp_discrepancy:xmax':1000,
    'sharp_discrepancy:seed':30,
    'sharp_discrepancy:itermax':5,
    'discrepancy:xmax':500,
    'discrepancy:ymax':500,
    'discrepancy:zmax':500,
    'discrepancy:nmax':2000}

    tf_param = {'epochs': 128,
    'batch_size':16,
    'validation_split':0.1,
    'loss':'mse',
    'optimizer':'adam',
    'layers':[8,64,64,64,1],
    'activation':['relu','relu','relu','relu','linear'],
    'metrics':['mse']}

    rbf_param = {'function': 'gaussian',
    'epsilon':None,
    'smooth':0.,
    'norm':'euclidean'}


    main_test(**{**kwargs,**tf_param,**rbf_param})

