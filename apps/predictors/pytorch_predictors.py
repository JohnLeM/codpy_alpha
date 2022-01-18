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


class PytorchRegressor(data_predictor):

    def get_params(**kwargs): return kwargs.get('PytorchRegressor',{})

    def get_model(self, **kwargs):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        self.epochs = kwargs.get('epochs',500)
        batch_size = kwargs.get('batch_size',128)
        layers = kwargs.get('layers', [128,64])
        out_layer =  kwargs.get('out_layer', [1])
        self.loss = kwargs.get('loss',nn.CrossEntropyLoss())
        activation = kwargs.get('activation',nn.ReLU())
        dataset=TensorDataset(torch.tensor(get_matrix(self.x),dtype=torch.float),torch.tensor(get_matrix(self.fx),dtype=torch.float))
        self.dataloader=DataLoader(dataset,batch_size = batch_size, shuffle=False)
        return PytorchNet(input_dim = self.D ,hidden_dim = layers, out_dim = out_layer, activation = activation, printnet=0)

    def predictor(self,**kwargs):
        import torch
        params = PytorchRegressor.get_params(**kwargs)
        model = self.get_model(**params)
        optimizer = params.get('optimizer', torch.optim.Adam)
        optimizer = optimizer(PytorchNet.parameters(model), lr=0.001)
        model.fit(epochs = self.epochs, dataloader = self.dataloader, model = model, optimizer = optimizer, loss = self.loss)
        z = torch.FloatTensor(get_matrix(self.z))
        self.f_z = model.predict(z)

    def id(self,name = "Pytorch"):
        return "Pytorch"


class PytorchClassifier(codpyprClassifier, add_confusion_matrix):

    def get_params(**kwargs): return kwargs.get('PytorchClassifier',{})

    def get_model(self, **kwargs):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        import os
        params = PytorchClassifier.get_params(**kwargs)
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        self.epochs = params.get('epochs',5)
        batch_size = params.get('batch_size',128)
        layers = params.get('layers', [128,64])
        out_layer =  params.get('out_layer', [10])
        self.loss = params.get('loss',nn.CrossEntropyLoss())
        activation = params.get('activation',nn.ReLU())
        data_type = params.get("datatype", "float")
        dataset=TensorDataset(torch.tensor(self.x,dtype=torch.float),torch.tensor(self.fx,dtype=torch.long))
        self.dataloader=DataLoader(dataset,batch_size = batch_size, shuffle=False)
        return PytorchNet(input_dim = self.D ,hidden_dim = layers, out_dim = out_layer, activation = activation, printnet=0)

    def predictor(self,**kwargs):
        import torch
        optimizer = kwargs.get('optimizer', torch.optim.Adam)
        model = self.get_model(**kwargs)
        optimizer = optimizer(PytorchNet.parameters(model), lr=0.001)
        model.fit(epochs = self.epochs, dataloader = self.dataloader, model = model, optimizer = optimizer, loss = self.loss)
        z = torch.FloatTensor(self.z)
        self.f_z = model.predict_labeled(z, **kwargs)

    def id(self,name = "Pytorch"):
        return "Pytorch"

class PytorchNet(nn.Module):
    def __init__(self,input_dim, hidden_dim, out_dim, activation, printnet = False):
        super(PytorchNet, self).__init__()
        dropout = 0.2
        current_dim = input_dim
        layers = []
        for hdim in hidden_dim:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(activation)
            #layers.append(nn.Dropout(p = dropout))
            current_dim = hdim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)
        if printnet == True:
            print(self.net)

    def fit(self, epochs, dataloader, model, optimizer, loss):
        self.model = model
        for epoch in range(epochs):
            Loss=None
            for batch_x,batch_fx in dataloader:
                fx_predict = model(batch_x)
                Loss = loss(fx_predict, batch_fx)
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()
            if (epoch + 1)%100==0:
               print("step: {0} , loss: {1}".format(epoch+1,Loss.item()))
    
    def predict(self, z):        
        f_z = self.model(z).clone().detach().numpy()
        return f_z

    def predict_labeled(self, z, **kwargs): 
        get_proba = kwargs.get('get_proba',False)
        f_z = []
        with torch.no_grad():
            if get_proba:
                for i,data in enumerate(z):
                    y_pred = self.model(data).clone().detach().numpy()
                    f_z.append(y_pred)
                f_z = np.array(f_z)
            else:
                for i,data in enumerate(z):
                    y_pred = self.model(data)
                    f_z.append(y_pred.argmax().item())
                f_z = np.asarray(f_z).reshape((len(f_z), 1))
        return f_z


    def forward(self, input:torch.FloatTensor):
        return self.net(input)


    


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


    torch_param = {'epochs': 100,
    'layers': [10,100,100,50],
    'loss': nn.MSELoss(),
    'batch_size': 10,
    'loss': nn.MSELoss(),
    'activation': nn.ReLU(),
    'optimizer': torch.optim.Adam,
    'out_layer': 1}

    
    scenarios_list = [ (1, 100*i, 50,100*i ) for i in np.arange(1,5,1)]
    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    PytorchRegressor(set_kernel = set_per_kernel),
    data_accumulator(), **torch_param)
    results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
    print(results)
    list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 2)
    pass


