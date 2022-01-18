# from IPython import get_ipython
# try:
#     get_ipython().run_line_magic('tensorflow_version', '1.x')
#     get_ipython().run_line_magic('matplotlib', 'inline')
# except Exception:
#     pass

# import tensorflow as tf
# print(tf.__version__)
# print(tf.test.is_gpu_available())

# # we want TF 1.x
# assert tf.__version__ < "2.0"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)
print(tf.test.is_gpu_available())


# disable annoying warnings
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# import other useful libs
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error
import statistics as stat

import os, sys
from pathlib import Path

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:sys.path.append(parentdir)
from common_include import *
from data_generators import *
from predictors import *



set_matern_norm_kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_norm_kernel, 0,1e-8 ,map_setters.set_standard_mean_map)
set_sampler_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,1e-8 ,map_setters.set_unitcube_map)
set_per_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)
set_gaussian_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel,2,1e-8,map_setters.set_standard_mean_map)
set_linear_regressor_kernel = kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, polynomial_order= 2, regularization = 1e-8, set_map = map_setters.set_unitcube_map)
set_tensornorm_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,1e-8 ,None)


# representation of real numbers in TF, change here for 32/64 bits
real_type = tf.float32
# real_type = tf.float64

def vanilla_net(
    input_dim,      # dimension of inputs, e.g. 10
    hidden_units,   # units in hidden layers, assumed constant, e.g. 20
    hidden_layers,  # number of hidden layers, e.g. 4
    seed):          # seed for initialization or None for random
    
    # set seed
    tf.set_random_seed(seed)
    
    # input layer
    xs = tf.placeholder(shape=[None, input_dim], dtype=real_type)
    
    # connection weights and biases of hidden layers
    ws = [None]
    bs = [None]
    # layer 0 (input) has no parameters
    
    # layer 0 = input layer
    zs = [xs] # eq.3, l=0
    
    # first hidden layer (index 1)
    # weight matrix
    ws.append(tf.get_variable("w1", [input_dim, hidden_units],         initializer = tf.variance_scaling_initializer(), dtype=real_type))
    # bias vector
    bs.append(tf.get_variable("b1", [hidden_units],         initializer = tf.zeros_initializer(), dtype=real_type))
    # graph
    zs.append(zs[0] @ ws[1] + bs[1]) # eq. 3, l=1
    
    # second hidden layer (index 2) to last (index hidden_layers)
    for l in range(1, hidden_layers): 
        ws.append(tf.get_variable("w%d"%(l+1), [hidden_units, hidden_units],             initializer = tf.variance_scaling_initializer(), dtype=real_type))
        bs.append(tf.get_variable("b%d"%(l+1), [hidden_units],             initializer = tf.zeros_initializer(), dtype=real_type))
        zs.append(tf.nn.softplus(zs[l]) @ ws[l+1] + bs[l+1]) # eq. 3, l=2..L-1

    # output layer (index hidden_layers+1)
    ws.append(tf.get_variable("w"+str(hidden_layers+1), [hidden_units, 1],             initializer = tf.variance_scaling_initializer(), dtype=real_type))
    bs.append(tf.get_variable("b"+str(hidden_layers+1), [1],         initializer = tf.zeros_initializer(), dtype=real_type))
    # eq. 3, l=L
    zs.append(tf.nn.softplus(zs[hidden_layers]) @ ws[hidden_layers+1] + bs[hidden_layers+1]) 
    
    # result = output layer
    ys = zs[hidden_layers+1]
    
    # return input layer, (parameters = weight matrices and bias vectors), 
    # [all layers] and output layer
    return xs, (ws, bs), zs, ys

# compute d_output/d_inputs by (explicit) backprop in vanilla net
def backprop(
    weights_and_biases, # 2nd output from vanilla_net() 
    zs):                # 3rd output from vanilla_net()
    
    ws, bs = weights_and_biases
    L = len(zs) - 1
    
    # backpropagation, eq. 4, l=L..1
    zbar = tf.ones_like(zs[L]) # zbar_L = 1
    for l in range(L-1, 0, -1):
        zbar = (zbar @ tf.transpose(ws[l+1])) * tf.nn.sigmoid(zs[l]) # eq. 4
    # for l=0
    zbar = zbar @ tf.transpose(ws[1]) # eq. 4
    
    xbar = zbar # xbar = zbar_0
    
    # dz[L] / dx
    return xbar    

# combined graph for valuation and differentiation
def twin_net(input_dim, hidden_units, hidden_layers, seed):
    
    # first, build the feedforward net
    xs, (ws, bs), zs, ys = vanilla_net(input_dim, hidden_units, hidden_layers, seed)
    
    # then, build its differentiation by backprop
    xbar = backprop((ws, bs), zs)
    
    # return input x, output y and differentials d_y/d_z
    return xs, ys, xbar

# %%
def vanilla_training_graph(input_dim, hidden_units, hidden_layers, seed):
    
    # net
    inputs, weights_and_biases, layers, predictions =         vanilla_net(input_dim, hidden_units, hidden_layers, seed)
    
    # backprop even though we are not USING differentials for training
    # we still need them to predict derivatives dy_dx 
    derivs_predictions = backprop(weights_and_biases, layers)
    
    # placeholder for labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    
    # loss 
    loss = tf.losses.mean_squared_error(labels, predictions)
    
    # optimizer
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    # return all necessary 
    return inputs, labels, predictions, derivs_predictions, learning_rate, loss, optimizer.minimize(loss)

# training loop for one epoch
def vanilla_train_one_epoch(# training graph from vanilla_training_graph()
                            inputs, labels, lr_placeholder, minimizer,   
                            # training set 
                            x_train, y_train,                           
                            # params, left to client code
                            learning_rate, batch_size, session):        
    
    m, n = x_train.shape
    
    # minimization loop over mini-batches
    first = 0
    last = min(batch_size, m)
    while first < m:
        session.run(minimizer, feed_dict = {
            inputs: x_train[first:last], 
            labels: y_train[first:last],
            lr_placeholder: learning_rate
        })
        first = last
        last = min(first + batch_size, m)

def diff_training_graph(
    # same as vanilla
    input_dim, 
    hidden_units, 
    hidden_layers, 
    seed, 
    # balance relative weight of values and differentials 
    # loss = alpha * MSE(values) + beta * MSE(greeks, lambda_j) 
    # see online appendix
    alpha, 
    beta,
    lambda_j):
    
    # net, now a twin
    inputs, predictions, derivs_predictions = twin_net(input_dim, hidden_units, hidden_layers, seed)
    
    # placeholder for labels, now also derivs labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    derivs_labels = tf.placeholder(shape=[None, derivs_predictions.shape[1]], dtype=real_type)
    
    # loss, now combined values + derivatives
    loss = alpha * tf.losses.mean_squared_error(labels, predictions)     + beta * tf. losses.mean_squared_error(derivs_labels * lambda_j, derivs_predictions * lambda_j)
    
    # optimizer, as vanilla
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    # return all necessary tensors, including derivatives
    # predictions and labels
    return inputs, labels, derivs_labels, predictions, derivs_predictions,             learning_rate, loss, optimizer.minimize(loss)

def diff_train_one_epoch(inputs, labels, derivs_labels, 
                         # graph
                         lr_placeholder, minimizer,             
                         # training set, extended
                         x_train, y_train, dydx_train,          
                         # params
                         learning_rate, batch_size, session):   
    
    m, n = x_train.shape
    
    # minimization loop, now with Greeks
    first = 0
    last = min(batch_size, m)
    while first < m:
        session.run(minimizer, feed_dict = {
            inputs: x_train[first:last], 
            labels: y_train[first:last],
            derivs_labels: dydx_train[first:last],
            lr_placeholder: learning_rate
        })
        first = last
        last = min(first + batch_size, m)
def train(description,
          # neural approximator
          approximator,              
          # training params
          reinit=True, 
          epochs=100, 
          # one-cycle learning rate schedule
          learning_rate_schedule=[    (0.0, 1.0e-8), \
                                      (0.2, 0.1),    \
                                      (0.6, 0.01),   \
                                      (0.9, 1.0e-6), \
                                      (1.0, 1.0e-8)  ], 
          batches_per_epoch=16,
          min_batch_size=256,
          # callback function and when to call it
          callback=None,           # arbitrary callable
          callback_epochs=[]):     # call after what epochs, e.g. [5, 20]
              
    # batching
    batch_size = max(min_batch_size, approximator.m // batches_per_epoch)
    
    # one-cycle learning rate sechedule
    lr_schedule_epochs, lr_schedule_rates = zip(*learning_rate_schedule)
            
    # reset
    if reinit:
        approximator.session.run(approximator.initializer)
    
    # callback on epoch 0, if requested
    if callback and 0 in callback_epochs:
        callback(approximator, 0)
        
    # loop on epochs, with progress bar (tqdm)
    for epoch in tqdm_notebook(range(epochs), desc=description):
        
        # interpolate learning rate in cycle
        learning_rate = np.interp(epoch / epochs, lr_schedule_epochs, lr_schedule_rates)
        
        # train one epoch
        
        if not approximator.differential:
        
            vanilla_train_one_epoch(
                approximator.inputs, 
                approximator.labels, 
                approximator.learning_rate, 
                approximator.minimizer, 
                approximator.x, 
                approximator.y, 
                learning_rate, 
                batch_size, 
                approximator.session)
        
        else:
        
            diff_train_one_epoch(
                approximator.inputs, 
                approximator.labels, 
                approximator.derivs_labels,
                approximator.learning_rate, 
                approximator.minimizer, 
                approximator.x, 
                approximator.y, 
                approximator.dy_dx,
                learning_rate, 
                batch_size, 
                approximator.session)
        
        # callback, if requested
        if callback and epoch in callback_epochs:
            callback(approximator, epoch)

    # final callback, if requested
    if callback and epochs in callback_epochs:
        callback(approximator, epochs)        


epsilon = 1.0e-08
def normalize_data(x_raw, y_raw, dydx_raw=None, crop=None):
    
    # crop dataset
    m = crop if crop is not None else x_raw.shape[0]
    x_cropped = x_raw[:m]
    y_cropped = y_raw[:m]
    dycropped_dxcropped = dydx_raw[:m] if dydx_raw is not None else None
    
    # normalize dataset
    x_mean = x_cropped.mean(axis=0)
    x_std = x_cropped.std(axis=0) + epsilon
    x = (x_cropped- x_mean) / x_std
    y_mean = y_cropped.mean(axis=0)
    y_std = y_cropped.std(axis=0) + epsilon
    y = (y_cropped-y_mean) / y_std
    
    # normalize derivatives too
    if dycropped_dxcropped is not None:
        dy_dx = dycropped_dxcropped / y_std * x_std 
        # weights of derivatives in cost function = (quad) mean size
        lambda_j = 1.0 / np.sqrt((dy_dx ** 2).mean(axis=0)).reshape(1, -1)
    else:
        dy_dx = None
        lambda_j = None
    
    return x_mean, x_std, x, y_mean, y_std, y, dy_dx, lambda_j


class Neural_Approximator():
    
    def __init__(self, x_raw, y_raw, 
                 dydx_raw=None):      # derivatives labels, 
       
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw
        
        # tensorflow logic
        self.graph = None
        self.session = None
                        
    def __del__(self):
        if self.session is not None:
            self.session.close()
        
    def build_graph(self,
                differential,       # differential or not           
                lam,                # balance cost between values and derivs  
                hidden_units, 
                hidden_layers, 
                weight_seed):
        
        # first, deal with tensorflow logic
        if self.session is not None:
            self.session.close()

        self.graph = tf.Graph()
        
        with self.graph.as_default():
        
            # build the graph, either vanilla or differential
            self.differential = differential
            
            if not differential:
            # vanilla 
                
                self.inputs,                 self.labels,                 self.predictions,                 self.derivs_predictions,                 self.learning_rate,                 self.loss,                 self.minimizer                 = vanilla_training_graph(self.n, hidden_units, hidden_layers, weight_seed)
                    
            else:
            # differential
            
                if self.dy_dx is None:
                    raise Exception("No differential labels for differential training graph")
            
                self.alpha = 1.0 / (1.0 + lam * self.n)
                self.beta = 1.0 - self.alpha
                
                self.inputs,                 self.labels,                 self.derivs_labels,                 self.predictions,                 self.derivs_predictions,                 self.learning_rate,                 self.loss,                 self.minimizer = diff_training_graph(self.n, hidden_units,                                                      hidden_layers, weight_seed,                                                      self.alpha, self.beta, self.lambda_j)
        
            # global initializer
            self.initializer = tf.global_variables_initializer()
            
        # done
        self.graph.finalize()
        self.session = tf.Session(graph=self.graph)
                        
    # prepare for training with m examples, standard or differential
    def prepare(self, 
                m, 
                differential,
                lam=1,              # balance cost between values and derivs  
                # standard architecture
                hidden_units=20, 
                hidden_layers=4, 
                weight_seed=3549):

        # prepare dataset
        self.x_mean, self.x_std, self.x, self.y_mean, self.y_std, self.y, self.dy_dx, self.lambda_j =             normalize_data(self.x_raw, self.y_raw, self.dydx_raw, m)
        
        # build graph        
        self.m, self.n = self.x.shape        
        self.build_graph(differential, lam, hidden_units, hidden_layers, weight_seed)
        
    def train(self,            
              description="training",
              # training params
              reinit=True, 
              epochs=100, 
              # one-cycle learning rate schedule
              learning_rate_schedule=[
                  (0.0, 1.0e-8), 
                  (0.2, 0.1), 
                  (0.6, 0.01), 
                  (0.9, 1.0e-6), 
                  (1.0, 1.0e-8)], 
              batches_per_epoch=16,
              min_batch_size=256,
              # callback and when to call it
              # we don't use callbacks, but this is very useful, e.g. for debugging
              callback=None,           # arbitrary callable
              callback_epochs=[]):     # call after what epochs, e.g. [5, 20]
              
        train(description, 
              self, 
              reinit, 
              epochs, 
              learning_rate_schedule, 
              batches_per_epoch, 
              min_batch_size,
              callback, 
              callback_epochs)
     
    def predict_values(self, x):
        # scale
        x_scaled = (x-self.x_mean) / self.x_std 
        # predict scaled
        y_scaled = self.session.run(self.predictions, feed_dict = {self.inputs: x_scaled})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        return y

    def predict_values_and_derivs(self, x):
        # scale
        x_scaled = (x-self.x_mean) / self.x_std
        # predict scaled
        y_scaled, dyscaled_dxscaled = self.session.run(
            [self.predictions, self.derivs_predictions], 
            feed_dict = {self.inputs: x_scaled})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        dydx = self.y_std / self.x_std * dyscaled_dxscaled
        return y, dydx
# helper analytics    
def bsPrice(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)
    return spot * norm.cdf(d1) - strike * norm.cdf(d2)

def bsDelta(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    return norm.cdf(d1)

def bsVega(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + vol * vol * T) / vol / np.sqrt(T)
    return spot * np.sqrt(T) * norm.pdf(d1)
    
# The code below trains approximators on LSM samples simulated by our *BlackScholes* class. Classical deep learning is able to learn very accurate approximations in this simple case, so differential learning doesn't make much difference, although it improves the learned shape and differentials on small datasets.
# helper analytics
def bachPrice(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return  vol * np.sqrt(T) * norm.pdf(d) + (spot - strike) * norm.cdf(d)

def bachDelta(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return norm.cdf(d)

def bachVega(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return np.sqrt(T) * norm.pdf(d)
#
    
# generates a random correlation matrix
def genCorrel(n,seed = None):
    if (seed): np.random.seed(seed)
    randoms = np.random.uniform(low=-1., high=1., size=(2*n, n))
    cov = randoms.T @ randoms
    invvols = np.diag(1. / np.sqrt(np.diagonal(cov)))
    return np.linalg.multi_dot([invvols, cov, invvols])

def gena(n,seed = None):
    if (seed): np.random.seed(seed)
    a = np.random.uniform(low=1., high=10., size=n)
    a /= np.sum(a)
    return a
def genVols(n,seed = None):
    if (seed): np.random.seed(seed)
    return np.random.uniform(low=5., high = 50., size = n)

class Bachelier:
    seed1, seed2,seed3 = 42, 75, 35
    seedparam = 13

    def set_data(self,n,T1=1,T2=2,K=1.10,bktVol=0.2,**kwargs):
        self.bktVol = bktVol
        self.seed1=kwargs.get('seed1', self.seed1)
        self.seed2=kwargs.get('seed2', self.seed2)
        self.seed3=kwargs.get('seed3', self.seed3)

        self.n = n
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.x, self.fx,self.dfx,self.z,self.fz, self.dfz,self.xAxis = [],[],[],[],[],[],[]
        if n==0:return
        # spots all currently 1, without loss of generality
        self.S0 = np.repeat(1., self.n)
        # random weights
        self.a = gena(self.n,self.seedparam)

        self.correlation = self.get_correlation(self.bktVol)

        #print("vol basket :",self.bktVol)
        #print("vol underlying :",self.correlation)
        # Choleski etc. for simulation

    def get_normals(self, N, D, seed=None): 
        if N*D==0: return
        if (seed): np.random.seed(seed)
        out = np.random.normal(size=[N * D]).reshape((N,D))
        # plt.scatter(out[:,0],out[:,1],color="red")
        return out
    def get_correlation(self, bktVol): 
        #random correls, but normalization to bktvol
        corr = genCorrel(self.n,self.seedparam)
        vols = genVols(self.n,self.seedparam)
        diagv = np.diag(vols)
        cov = np.linalg.multi_dot([diagv, corr, diagv])
        correlation = np.linalg.cholesky(cov)
        tempvol = np.sqrt(np.linalg.multi_dot([self.a.T, cov, self.a]))
        cov *= bktVol * bktVol / ( tempvol * tempvol)
        correlation *= bktVol / tempvol
        testbktvol = np.sqrt(np.linalg.multi_dot([self.a.T, cov, self.a])) 

        return correlation

    def variables(self, time, m, x0=[], seed=None): 
        if m==0: return
        if (seed): np.random.seed(seed)
        # normals = self.get_normals(m , self.n, seed = seed)
        normals = np.random.normal(size=[m * self.n]).reshape((m,self.n))
        # print("covar:",np.cov(normals.T))
        inc = np.sqrt(time) * normals @ self.correlation.T
        if len(x0)==0: x0 = np.asarray([self.S0 for n in range(0,m)])
        x1 = x0 + inc
        return np.asarray(x1)
          
    def basket(self, x, time = 2.): 
        if len(x) == 0: return
        if type(x) == type([]): return [self.basket(s,time) for s in x]
        return np.dot(x, self.a)
    def values(self, x, time = 2.): 
        if len(x) == 0: return
        if type(x) == type([]): return [self.values(s,time) for s in x]
        bkt = self.basket(x, time)
        pay = np.maximum(0, bkt - self.K)
        return np.asarray(pay).reshape(-1,1)
    def nabla_values(self, x, time = 2.): 
        if len(x) == 0: return
        if type(x) == type([]): return [self.nabla_values(s,time) for s in x]
        bkt = self.basket(x,time)
        Z =  np.where(bkt > self.K, 1.0, 0.0).reshape((-1,1)) * self.a.reshape((1,-1))
        return Z

    def refvalues(self, x, maturity): 
        return bachPrice(self.basket(x), self.K, self.bktVol, maturity).reshape((-1, 1))

    
    def trainingSet(self, m, seed1=None,seed2=None): 
        if m==0 : return
        if seed1==None: seed1 = self.seed1
        if seed2==None: seed2 = self.seed2
        S1 = self.variables(self.T1,m,seed=seed1)
        S2 = self.variables(self.T2-self.T1,m,S1,seed2)
        # S2 = self.variables(self.T2,m,seed=seed2)
        bkt2 = self.basket(S2)
        payoff = self.values(S2).reshape(-1,1)
        nabla_payoff = self.nabla_values(S2)
        # [S1, payoff, nabla_payoff, S2], bkt2, permutation = lexicographical_permutation(bkt2,[S1, payoff, nabla_payoff, S2])
        return S1, payoff, nabla_payoff, S2
    
    def testSet(self, m, seed1=None):
        if seed1==None: seed1 = self.seed3
        spots = self.variables(self.T1,m,seed=seed1)
        baskets = self.basket(spots)
        prices = bachPrice(baskets, self.K, self.bktVol, self.T2 - self.T1).reshape((-1, 1))
        deltas = bachDelta(baskets, self.K, self.bktVol, self.T2 - self.T1).reshape((-1, 1))
        deltas = deltas@self.a.reshape((1, -1))
        vegas = bachVega(baskets, self.K, self.bktVol, self.T2 - self.T1) 
        return spots, prices, deltas, vegas 

    def __init__(self,n,T1=1,T2=2,K=1.10,bktVol=0.2,**kwargs):
        self.set_data(n=n,T1=T1,T2=T2,K=K,bktVol=bktVol,**kwargs)


class data_generator_Bachelier(data_generator,Bachelier):
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        Bachelier.set_data(self,n =D,**kwargs)
        if (D*Nx*Ny*Nz >0):
            x  = self.variables(time = self.T1,m=Nx,seed = self.seed1)
            x2  = self.variables(time = self.T2 - self.T1, m = Nx, x0=x, seed=self.seed2)
            fx  = self.values(x2).reshape((-1, 1))
            dfx  = self.nabla_values(x2,time = self.T2)
            z  = self.variables(time = self.T1,m=Nz,seed = self.seed3)
            fz  = self.refvalues(z, maturity = self.T2 - self.T1)

            if (Ny < Nx): 
                indices = np.arange(start = 0,stop = Nx,step = Nx/Ny).astype(int)
                fx,x,permutation = lexicographical_permutation(fx,x)
                y,fy = x[indices], fx[indices]
                Ny = len(fy)

                # probas = np.array(np.arange(start = .5/Ny,stop = 1.,step = 1./Ny)).reshape(Ny,1)
                # y,fy, permutation = alg.iso_probas_projection(x = x, fx = fx, probas = probas, set_codpy_kernel = set_tensornorm_kernel, rescale = True)
                debug = 1.


                # check projection vareiable set and projection values set
                # plotfx,plotx,permutation = lexicographical_permutation(fy2.flatten())
                # plt.plot(plotx,plotfx)
                # plt.show()

                # check projection set reordering
                # x,y = y,y2
                # plt.plot([y[0:N,0], x[0:N,0]],[y[0:N,1], x[0:N,1]])
                # plt.show()
            else:
                Ny = Nx
                y = x.copy()
                fy = fx.copy()

            #print("mean(fx):",stat.mean(fx.flatten()),", mean(fz):", stat.mean(fz.flatten()) )
        return  x, fx, y, fy, z, fz
    def copy(self):
        return self.copy_data(data_generator_Bachelier())

class data_generator_Bachelier_iid(data_generator_Bachelier):
    def set_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        super().set_data(D,Nx,Ny,Nz,**kwargs)
        if (self.D*self.Nx*self.Ny*self.Nz >0):
            self.x2  = self.variables(time = self.T2, m = Nx, seed=self.seed2)
            self.fx  = self.values(self.x2).reshape((-1, 1))
            if self.seed1 == self.seed3 and self.Nx == self.Nz :
                self.z  = self.x
                self.fz  = self.refvalues(self.z, maturity = self.T2 - self.T1)
    def id(self,name = "i.i.d."):
        return name


            #print("mean(fx):",stat.mean(self.fx.flatten()),", mean(fz):", stat.mean(self.fz.flatten()) )
            
class data_generator_Bachelier_sharp(data_generator_Bachelier_iid):
    set_kernel = set_sampler_kernel
    # set_kernel = set_sampler_kernel
    def set_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        super().set_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz,**kwargs)
        self.set_kernel()
        # set_tensornorm_kernel()

    def variables(self, time, m, x0=[], seed=None): 
        if m==0: return
        if (seed): np.random.seed(seed)
        normals = self.get_normals(m , self.n, seed = seed)
        # print("covar:",np.cov(normals.T))
        inc = np.sqrt(time) * normals @ self.correlation.T
        if len(x0)==0: x0 = np.asarray([self.S0 for n in range(0,m)])
        x1 = x0 + inc
        return np.asarray(x1)

    normals_ = np.empty(0)
    def get_normals(self, N, D, seed=None): 
        from scipy import special
        if N*D==0: return
        if self.normals_.shape != (N,D):
            self.normals_ = alg.get_normals(N=N, D=D, nmax = 50, set_codpy_kernel = self.set_kernel)
            #plt.scatter(self.normals_[:,0],self.normals_[:,1],color="green")
            #test = super().get_normals(N=N, D=D, seed = seed)
            #plt.scatter(test[:,0],test[:,1],color="red")
            #plt.show()
        return self.normals_

    def get_correlation(self, bktVol): 
        #random correls, but normalization to bktvol
        import scipy
        corr = genCorrel(self.n,self.seedparam)
        vols = genVols(self.n,self.seedparam)
        diagv = np.diag(vols)
        cov = np.linalg.multi_dot([diagv, corr, diagv])
        correlation = scipy.linalg.sqrtm(cov)
        tempvol = np.sqrt(np.linalg.multi_dot([self.a.T, cov, self.a]))
        cov *= bktVol * bktVol / ( tempvol * tempvol)
        correlation *= bktVol / tempvol
        testbktvol = np.sqrt(np.linalg.multi_dot([self.a.T, cov, self.a])) 

        return correlation
    def id(self,name = "sharp"):
        return name



class NN_predictor_standard(data_predictor):
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz):
        # print("initializing neural approximator")
            regressor = Neural_Approximator(x_raw = self.x, y_raw = self.fx, dydx_raw=None)
            regressor.prepare(self.Nx, False, hidden_units=int(self.Ny / 4))
            regressor.train("standard training")
            self.f_z, self.df_z = regressor.predict_values_and_derivs(self.z)
            # plotD(BachelierGenerator.basket(x = self.x),self.fx,BachelierGenerator.basket(x= self.z),self.f_z)
    def copy(self):
        return self.copy_data(NN_predictor_standard())

    def id(self,name = "ANN"):
        return name 

class NN_predictor_differential(NN_predictor_standard):
    def set_data(self,generator):
        super().set_data(generator)
        if (self.D*self.Nx*self.Ny*self.Nz ):
            start = time.time()
            # print("initializing neural approximator")
            regressor = Neural_Approximator(self.x, self.fx, self.dfx)
            regressor.prepare(self.Nz, True, hidden_units=int(self.Ny / 4),weight_seed=generator.seed3)
            regressor.train("differential training")
            self.f_z, self.df_z = regressor.predict_values_and_derivs(self.z)
            self.elapsed_predict_time = time.time()-start

    def copy(self):
        return self.copy_data(NN_predictor_differential())
    def id(self,name = "ANNDiff"):
        return name 

class Linear_regressor_predictor(data_predictor):
    def set_data(self,generator):
        super().set_data(generator)
        if (self.D*self.Nx*self.Ny*self.Nz ):
            start = time.time()
            self.f_z=op.projection(self.x,self.y,self.z,self.fx,set_codpy_kernel = set_linear_regressor_kernel,rescale = True)
            self.elapsed_predict_time = time.time()-start
            self.norm_function = op.norm(self.x,self.y,self.z,self.fx,set_codpy_kernel = None,rescale = False)
            self.df_z = []
            # if (len(x) < 1025): self.discrepancy_errors = op.discrepancy(x=x,y=y,z=z,set_codpy_kernel = None,rescale = False)
    def copy(self):
        return self.copy_data(Linear_regressor_predictor())
    def id(self,name = "LinReg"):
        return name 

class Pi_predictor(data_predictor):
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            # x2=BachelierGenerator.variables(BachelierGenerator.T2-BachelierGenerator.T1,Nx,generator.x2,seed=BachelierGenerator.seed2)
            #self.f_z=op.projection(self.x,self.y,self.x,self.fx,set_codpy_kernel = set_tensornorm_kernel,rescale = True)
            self.f_z=alg.Pi(x = self.x, z = self.generator.x2, fz = self.fx, nmax=100,set_codpy_kernel = self.set_kernel)
            # print("mean(x[:,0]):",stat.mean(self.x[:,1]),"mean(x[:,1]):",stat.mean(self.x[:,1]),"mean(x2[:,0]):",stat.mean(generator.x2[:,1]),"mean(x2[:,1]):",stat.mean(generator.x2[:,1]),", mean(fz):")
            # print("mean(fx):",stat.mean(self.fx.flatten()),", mean(fz):", stat.mean(generator.fz.flatten()),", mean(f_z):", stat.mean(self.f_z.flatten()) )
            self.f_z=op.projection(self.x,self.y,self.z,self.f_z,set_codpy_kernel = self.set_kernel,rescale = True)
            # print(", mean(f_z):", stat.mean(self.f_z.flatten()) )
            self.df_z = []
            # check projection vareiable set and projection values set
            # plotfx,plotx,permutation = lexicographical_permutation(generator.fy2.flatten())
            # plt.plot(plotx,plotfx)
            # plt.show()
            # check projection set reordering
            # x,y = self.y,generator.y2
            # plt.plot([y[:,0], x[:,0]],[y[:,1], x[:,1]])
            # plt.show()

    def copy(self):
        return self.copy_data(Pi_predictor())

    def id(self,name = "Pi:"):
        return name + self.generator.id()


class check_generator:
    def plot_ditribution(generator):
        self.x,self.y,self.z,self.fx,self.dfx,self.fz,self.dfz = generator.get_output_datas(Nx=100)
        multi_compare1D(x[permut[0:1000]],fx[permut[0:1000]],title = 'payoff vs underlying values',labelx='payoff values',labely='und:',figsizey=7,flip=False)

def permutation(arr,permut,axis=0):
    if axis ==0:return arr[permut]
    if axis ==arr.ndim-1:return arr[...,permut]
    return arr[...,permut,...]

def log_helper(x):
    return np.log2(x)

if __name__ == "__main__":
    codpy_param = {'rescale:xmax': 1000,
    'rescale:seed':42,
    'sharp_discrepancy:xmax':1000,
    'sharp_discrepancy:seed':30,
    'sharp_discrepancy:itermax':10,
    'discrepancy:xmax':500,
    'discrepancy:ymax':500,
    'discrepancy:zmax':500,
    'discrepancy:nmax':2000 
    }

    # fx = np.array([(1,9), (5,4), (1,0), (4,4), (3,0), (4,2), (4,1)],dtype=np.dtype([('x', int), ('y', int)]))

#use of scenario list instead
    scenarios_list = [ (2, 2**(i), 2**(i), 2**(i))  for i in np.arange(5,8,1)]
    scenarios = scenario_generator()
    data_generator_Bachelier_sharp_ = data_generator_Bachelier_sharp()
    scenarios.run_scenarios(scenarios_list,data_generator_Bachelier_sharp_,Pi_predictor(set_kernel = set_sampler_kernel),data_accumulator(),seed1 = 42, seed2 = 42, seed3 = 42, **codpy_param)
    basketxs = data_generator_Bachelier_sharp_.basket(x = scenarios.accumulator.get_xs())
    basketzs = data_generator_Bachelier_sharp_.basket(x = scenarios.accumulator.get_zs())
    ######outputs
    results = scenarios.accumulator.get_output_datas()
    print(results)
    # scenarios.accumulator.plot_learning_and_train_sets(basketxs,basketzs,labelx='Basket values',labely='')
    # scenarios.accumulator.plot_errors(basketzs,labelx='Basket values',labely='')
    scenarios.accumulator.plot_predicted_values(basketzs,labelx='Basket values',labely='')


    data_generator_Bachelier_iid_ = data_generator_Bachelier_iid()
    scenarios.run_scenarios(scenarios_list,data_generator_Bachelier_iid_,Pi_predictor(set_kernel = set_sampler_kernel),data_accumulator(),seed1 = 42, seed2 = 35, seed3 = 42, **codpy_param)
    basketxs = data_generator_Bachelier_iid_.basket(x = scenarios.accumulator.get_xs())
    basketzs = data_generator_Bachelier_iid_.basket(x = scenarios.accumulator.get_zs())
    ######outputs
    results = scenarios.accumulator.get_output_datas()
    print(results)
    # scenarios.accumulator.plot_learning_and_train_sets(basketxs,basketzs,labelx='Basket values',labely='')
    # scenarios.accumulator.plot_errors(basketzs,labelx='Basket values',labely='')
    scenarios.accumulator.plot_predicted_values(basketzs,labelx='Basket values',labely='')

    scenarios_list = [ (2, 2**i, 80, 2**i)  for i in np.arange(5,16,1)]
    # scenarios_list = [ (2, 2**(i-2), 2**(i-2), 4096)  for i in np.arange(6,11,1)]

    data_generator_Bachelier_ = data_generator_Bachelier(seed1 = 42, seed2 = 35, seed3 = 42)
    scenarios.run_scenarios(scenarios_list,data_generator_Bachelier_,codpyprRegressor(set_kernel = set_sampler_kernel),data_accumulator(),seed1 = 42, seed2 = 35, seed3 = 42, **codpy_param)
    basketxs = data_generator_Bachelier_.basket(x = scenarios.accumulator.get_xs())
    basketzs = data_generator_Bachelier_.basket(x = scenarios.accumulator.get_zs())
    results = scenarios.accumulator.get_output_datas()
    print(results)
    # scenarios.accumulator.plot_learning_and_train_sets(basketxs,basketzs,labelx='Basket values',labely='')
    # scenarios.accumulator.plot_errors(basketzs,labelx='Basket values',labely='')
    scenarios.accumulator.plot_predicted_values(basketzs,labelx='Basket values',labely='')


    scenarios.run_scenarios(scenarios_list,data_generator_Bachelier_,NN_predictor_standard(set_kernel = set_sampler_kernel),data_accumulator(),seed1 = 42, seed2 = 35, seed3 = 42, **codpy_param)
    results = scenarios.accumulator.get_output_datas()
    print(results)
    basketxs = data_generator_Bachelier_.basket(x = scenarios.accumulator.get_xs())
    basketzs = data_generator_Bachelier_.basket(x = scenarios.accumulator.get_zs())
    # scenarios.accumulator.plot_learning_and_train_sets(basketxs,basketzs,labelx='Basket values',labely='')
    # scenarios.accumulator.plot_errors(basketzs,labelx='Basket values',labely='')
    scenarios.accumulator.plot_predicted_values(basketzs,labelx='Basket values',labely='')

    scenarios.compare_plots(axis_field_labels = [("Nx","scores")],labelx='log2(Nx)',labely='scores',xscale ="log")
    scenarios.compare_plots(axis_field_labels = [("Nx","execution_time")],labelx='log2(Nx)',labely='scores',xscale ="log")