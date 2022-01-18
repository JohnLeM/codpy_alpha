import os, sys
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.append(parentdir)
data_path = os.path.dirname(__file__)
from common_include import * 
from data_generators import *
from codpy_predictors import *

def get_tensorflow_model(**kwargs):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    #tf.logging.set_verbosity(tf.logging.WARN)
    loss = kwargs.get('loss','mse')
    optimizer = kwargs.get('optimizer','adam')
    # layers = kwargs.get('layers',[8,64,64,1])
    layers = kwargs.get('layers',None)
    assert layers is not None
    activations = kwargs.get('activation',np.repeat('relu',len(layers)))
    metrics = kwargs.get('metrics',['mse'])
    input_shape = kwargs.get('input_shape',[1])

    model = tf.keras.Sequential()
    model.add (tf.keras.layers.Flatten(input_shape = input_shape))
    for layer,activation in zip(layers,activations):
        if len(activation): model.add(Dense(units = layer, activation=activation))
        else: model.add(Dense(units = layer))
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    return model


class tfRegressor(data_predictor):

    def get_params(**kwargs): return kwargs.get('tfRegressor',{})

    def predictor(self,**kwargs):
        model = get_tensorflow_model(input_shape = [self.D],**tfRegressor.get_params(**kwargs))
        epochs = kwargs.get('epochs',128)
        batch_size = kwargs.get('batch_size',16)
        validation_split = kwargs.get('validation_split',0.1)
        model.fit(self.x, self.fx, epochs=epochs,validation_split = validation_split,batch_size = batch_size, verbose=0)
        self.f_z = model.predict(self.z, verbose= 0)
    
    def id(self,name = "Tensorflow"):
        return "Tensorflow"

class tfClassifier(codpyprClassifier):


    def get_params(**kwargs): return kwargs.get('tfClassifier',{})

    def predictor(self,**kwargs):
        epochs = kwargs.get('epochs',5)
        get_proba = kwargs.get('get_proba',False)
        model = get_tensorflow_model(input_shape = [self.D],**tfClassifier.get_params(**kwargs))
        model.fit(self.x,self.fx,epochs=epochs, verbose=1)
        if get_proba:
            self.f_z = model.predict(self.z, verbose= 1)
        else:
            self.f_z = softmaxindice(model.predict(self.z, verbose= 1))
    def id(self,name = "Tensorflow"):
        return "Tensorflow"

def tf_test_unlabelled(scenarios_list,**kwargs): 

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
    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    tfRegressor(set_kernel = set_per_kernel),
    data_accumulator(), **kwargs)
    results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1)
    print(results)


def tf_test_mnist(scenarios_list,**kwargs): 
    pd_scenarios_list = pd.DataFrame(scenarios_list)
    set_mnist_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8 ,map_setters.set_mean_distance_map)
    scenarios = scenario_generator()
    scenarios.run_scenarios(scenarios_list,MNIST_data_generator(),tfClassifier(set_kernel = set_mnist_kernel),data_accumulator(),**kwargs)
    tf_results = scenarios.accumulator.get_output_datas().dropna(axis=1).T
    multi_plot([scenarios.predictor] ,add_confusion_matrix.plot_confusion_matrix)
    scenarios.accumulator.plot_confusion_matrices()


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

    kwargs['tfClassifier'] = {'epochs': 128,
    'batch_size':16,
    'validation_split':0.1,
    'loss':'sparse_categorical_crossentropy',
    'optimizer':'adam',
    'layers':[128,10],
    'activation':['relu','softmax'],
    'metrics':['accuracy']}

    scenarios_list = [ (784, 2**(i), 2**(i-2), 10000)  for i in np.arange(5,13,1)]
    tf_test_mnist(scenarios_list,**kwargs)
    pass

