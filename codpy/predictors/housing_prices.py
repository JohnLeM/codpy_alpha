import os,sys
import numpy as np
import pandas as pd
import time as time
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:sys.path.append(parentdir)
from common_include import *
from data_generators import * 
from predictors import * 

###initialization

def supervised_impl(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs):
    scenario_generator.run_scenarios(scenarios_list,generator,predictor,accumulator,**kwargs)
    if bool(kwargs.get("Show_results",True)):
        results = accumulator.get_output_datas().dropna(axis=1)
        print(results)
    if bool(kwargs.get("Show_confusion",False)):accumulator.plot_confusion_matrices(**kwargs,mp_title = "confusion matrices for "+predictor.id())
    if bool(kwargs.get("Show_clusters",False)):accumulator.plot_clusters(**kwargs, mp_title = "PCA clusters for "+predictor.id())
    if bool(kwargs.get("Show_clusters_index",False)):accumulator.plot_clusters(**kwargs, mp_title = "clusters for "+predictor.id(),index1=0,index2=1,xlabel = 'x',ylabel = 'y')
    if bool(kwargs.get("Show_maps_cluster",False)):print(accumulator.get_maps_cluster_indices())


class housing_codpy_extrapolator(codpyexRegressor):
    
    def predictor(self,**kwargs):
        self.variables_selector = variable_selector(self.x,self.y,self.z,self.fx,self.fz,error_fun = self.accuracy_score_function,**kwargs)
        kwargs['variables_cols_keep'] = self.variables_selector
        super().predictor(**kwargs)

    def new_method(self):
        pass
    def id(self,name = ""):
        return "housing codpy"

def housingprices_test(**kwargs):
    set_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2, 1e-8 ,map_setters.set_unitcube_map)
    # set_kernel = kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, polynomial_order= 2, regularization = 1e-8, set_map = map_setters.set_unitcube_map)
    # set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2, 1e-8 ,map_setters.set_mean_distance_map)
    # set_kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_tensor_kernel, 2,1e-8 ,map_setters.set_standard_mean_map)
    # set_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,1e-8 ,map_setters.set_unitcube_map)

#use of scenario list instead
    validator_compute=['accuracy_score','discrepancy_error','norm']
    scenario_generator_ = scenario_generator()
    # data_generator_ = housing_data_generator()
    data_generator_ = Boston_data_generator()
    x, fx, x, fx, z, fz = data_generator_.get_data(-1, -1, -1, -1)
    length_ = len(x)

    scenarios_list = [ (-1, i, i, -1)  for i in np.arange(length_,20,-(length_-20)/10) ]


    supervised_impl(scenario_generator_,scenarios_list,data_generator_,tfRegressor(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)
    
    supervised_impl(scenario_generator_,scenarios_list,data_generator_,GradientBoostingRegressor(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)
    
    supervised_impl(scenario_generator_,scenarios_list,data_generator_,AdaBoostRegressor(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,data_generator_,XGBRegressor(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,data_generator_,DecisionTreeRegressor(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,data_generator_,RandomForestRegressor(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,data_generator_,PytorchRegressor(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,data_generator_,SVR(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,data_generator_,housing_codpy_extrapolator(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
        **kwargs)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Nx","scores")],
        mp_title = "Benchmark methods",mp_ncols=1
    )

    scenario_generator_.compare_plots(
        axis_field_labels = [("Nx","discrepancy_errors")],
        mp_title = "Benchmark methods",mp_ncols=1
    )

    scenario_generator_.compare_plots(
        axis_field_labels = [("Nx","execution_time")],
        mp_title = "Benchmark methods",mp_ncols=1
    )

def get_params():
    import tensorflow as tf
    kwargs = {'rescale:xmax': 1000,
    'rescale:seed':42,
    'sharp_discrepancy:xmax':1000,
    'sharp_discrepancy:seed':30,
    'sharp_discrepancy:itermax':10,
    'discrepancy:xmax':500,
    'discrepancy:ymax':500,
    'discrepancy:zmax':500,
    'discrepancy:nmax':2000}

    kwargs['tfRegressor'] = {'epochs': 50,
    'batch_size':16,
    'validation_split':0.1,
    'loss':tf.keras.losses.mean_squared_error,
    'optimizer':tf.keras.optimizers.Adam(0.001),
    'layers':[8,64,64,1],
    'activation':['relu','relu','relu','linear'],
    'metrics':['mse']}

    kwargs['PytorchRegressor'] = {'epochs': 128,
    'layers': [8,64,64],
    'activation':['relu','linear'],
    'batch_size': 16,
    'loss': nn.MSELoss(),
    'activation': nn.ReLU(),
    'optimizer': torch.optim.Adam,
    "out_layer": 1}

    kwargs['XGBRegressor'] = {'epochs': 5,
    'max_depth': 3,
    'eta' : 0.3,
    'objective': 'multi:softmax',
    'num_class': 10,
    'num_boost_round':100}

    return kwargs
def main():
    import tensorflow as tf
    housingprices_test(**get_params())     


if __name__ == "__main__":
    main()
    pass    




