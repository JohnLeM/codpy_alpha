import os,sys
import numpy as np
import pandas as pd
import time as time
print(sys.path)
import codpy.codpy as cd
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:sys.path.append(parentdir)
from common_include import *
from data_generators import * 
from predictors import * 


###initialization

def show_mnist_picture(image):
    from matplotlib import pyplot as plt
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
def show_mnist_pictures(images, ax,**kwargs):
    ncols = kwargs.get("ncols",10)  
    max_images = kwargs.get("max_images",10)  
    numbers = min(len(images),max_images)
    ncols = min(ncols,numbers)
    if numbers == 0:return
    j = 0
    pixels = []
    for image in images:
        if len(pixels) == 0: pixels = image.reshape((28, 28))
        else: pixels = np.concatenate( (pixels,image.reshape((28, 28))),axis=1)
        j = j+1
        if j == numbers:break
    
    ax.imshow(pixels, cmap='gray')

def supervised_impl(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs):
    scenario_generator.run_scenarios(scenarios_list,generator,predictor,accumulator,**kwargs)
    if bool(kwargs.get("Show_results",True)):
        results = accumulator.get_output_datas().dropna(axis=1)
        print(results)
    if bool(kwargs.get("Show_confusion",False)):accumulator.plot_confusion_matrices(**kwargs,mp_title = "confusion matrices for "+predictor.id())
    if bool(kwargs.get("Show_clusters",False)):accumulator.plot_clusters(**kwargs, mp_title = "PCA clusters for "+predictor.id())
    if bool(kwargs.get("Show_clusters_index",False)):accumulator.plot_clusters(**kwargs, mp_title = "clusters for "+predictor.id(),index1=0,index2=1,xlabel = 'x',ylabel = 'y')
    if bool(kwargs.get("Show_maps_cluster",False)):print(accumulator.get_maps_cluster_indices())



def mnist_test(scenarios_list, **kwargs):
    set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8 ,map_setters.set_standard_min_map)
#use of scenario list instead
    validator_compute=['accuracy_score','discrepancy_error','norm']
    scenario_generator_ = scenario_generator()
    MNIST_data_generator_ = MNIST_data_generator()

    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,tfClassifier(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)
    
    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,GradientBoostingClassifier(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)
    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,AdaBoostClassifier(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)
    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,XGBClassifier(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,DecisionTreeClassifier(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)
    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,RandomForestClassifier(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,PytorchClassifier(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,SVC(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
       **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,codpyexClassifier(set_kernel = set_kernel),data_accumulator(),
        validator_compute = validator_compute,
        mp_max_items = 1,ncol=1,
        **kwargs)

    supervised_impl(scenario_generator_,scenarios_list,MNIST_data_generator_,codpyprClassifier(set_kernel = set_kernel),data_accumulator(),
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

if __name__ == "__main__":
    import tensorflow as tf
    scenarios_list = [ (784, 2**(i), 2**(i-2), 10000)  for i in np.arange(5,11,1)]

    kwargs = {'rescale:xmax': 1000,
    'rescale:seed':42,
    'sharp_discrepancy:xmax':1000,
    'sharp_discrepancy:seed':30,
    'sharp_discrepancy:itermax':10,
    'discrepancy:xmax':500,
    'discrepancy:ymax':500,
    'discrepancy:zmax':500,
    'discrepancy:nmax':2000,
    'Show_results':True,
    'Show_confusion':False,
    'Show_clusters':False,
    'Show_clusters_index':False,
    'Show_maps_cluster':False,
    }

    kwargs['tfClassifier'] = {'epochs': 128,
    'batch_size':16,
    'validation_split':0.1,
    'loss':'sparse_categorical_crossentropy',
    'optimizer':'adam',
    'layers':[128,10],
    'activation':['relu','softmax'],
    'metrics':['accuracy']}

    kwargs['PytorchClassifier'] = {'epochs': 128,
    'layers': [128],
    'batch_size': 16,
    'loss': nn.CrossEntropyLoss(),
    'activation': nn.ReLU(),
    'optimizer': torch.optim.Adam,
    "datatype": "long",
    "prediction": "labeled",
    "out_layer": 10}

    kwargs['XGBClassifier'] = {'epochs': 5,
    'max_depth': 3,
    'eta' : 0.3,
    'objective': 'multi:softmax',
    'num_class': 10,
    'num_boost_round':100}

    mnist_test(scenarios_list,**{**kwargs})     
    pass  