import os, sys
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.append(parentdir)
data_path = os.path.dirname(__file__)
from common_include import * 
from data_generators import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



####A standard run

def standard_supervised_run(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs):
    scenario_generator.run_scenarios(scenarios_list,generator,predictor,accumulator,**kwargs)
    if bool(kwargs.get("Show_results",True)):
        results = accumulator.get_output_datas().dropna(axis=1)
        print(results)
    if bool(kwargs.get("Show_confusion",True)):accumulator.plot_confusion_matrices(**kwargs,mp_title = "confusion matrices for "+predictor.id())
    if bool(kwargs.get("Show_maps",True)):print(accumulator.get_maps_cluster_indices())

######################### regressors ######################################""
class codpyprRegressor(data_predictor):
    
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            self.f_z = op.projection(x = self.x,y = self.y,z = self.z, fx = self.fx,set_codpy_kernel=self.set_kernel,rescale = True,**kwargs)
            pass
    def id(self,name = ""):
        return "codpy pred"

class codpyexRegressor(data_predictor):
    
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            self.f_z = op.projection(x = self.x,y = self.x,z = self.z, fx = self.fx,set_codpy_kernel=self.set_kernel,rescale = True,**kwargs)
    def id(self,name = ""):
        return "codpy extra"

############################ classifiers ###############################################
class codpyprClassifier(codpyprRegressor,add_confusion_matrix): #label_codpy_predictor
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if 'accuracy_score_function' not in kwargs: 
            from sklearn import metrics
            self.accuracy_score_function = metrics.accuracy_score

    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            get_proba = kwargs.get('get_proba',False)
            fx = unity_partition(self.fx)
            f_z = op.projection(x = self.x,y = self.y,z = self.z, fx = fx,set_codpy_kernel=self.set_kernel,rescale = True)
            if get_proba:
                self.f_z = f_z
            else:
                self.f_z = softmaxindice(f_z)
    def id(self,name = ""):
        return "codpy lab pred"
    def copy(self):
        return self.copy_data(codpyprClassifier())


class codpyexClassifier(codpyprClassifier):
    def copy(self):
        return self.copy_data(codpyexClassifier())
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            get_proba = kwargs.get('get_proba',False)
            fx = unity_partition(self.fx)
            f_z = op.projection(x = self.x,y = self.x,z = self.z, fx = fx,set_codpy_kernel=self.set_kernel,rescale = True)
            if get_proba:
                self.f_z = f_z
            else:
                self.f_z = softmaxindice(f_z)
    def id(self,name = ""):
        return "codpy lab extra"

################### Semi_supervised ######################################""
class codpyClusterClassifier(standard_cluster_predictor,add_confusion_matrix):
    def copy(self):
        return self.copy_data(codpyClusterClassifier())
    def predictor(self,**kwargs):
        self.y = alg.sharp_discrepancy(x = self.x, Ny=self.Ny,set_codpy_kernel = self.set_kernel, rescale = True,**kwargs)
        fx = unity_partition(fx = self.fx)
        debug = op.projection(x = self.x,y = self.y,z = self.z,fx = fx,set_codpy_kernel=self.set_kernel,rescale = True,**kwargs)
        self.f_z = softmaxindice(debug)
    def id(self,name = ""):
        return "codpy"

class codpyClusterPredictor(standard_cluster_predictor,add_confusion_matrix):
    def copy(self):
        return self.copy_data(codpyClusterPredictor())
    def predictor(self,**kwargs):
        self.y = alg.sharp_discrepancy(x = self.x, Ny=self.Ny,set_codpy_kernel = self.set_kernel, rescale = True,**kwargs)
        self.fx = alg.distance_labelling(self.x, self.y, set_codpy_kernel = None, rescale = True)
        if (self.x is self.z):
            self.f_z = self.fx
        else: 
            fx = unity_partition(fx = self.fx)
            debug = op.projection(x = self.x,y = self.y,z = self.z,fx = fx,set_codpy_kernel=self.set_kernel,rescale = True,**kwargs)
            self.f_z = softmaxindice(debug)
    def id(self,name = ""):
        return "codpy"        


def test_predictor(my_fun):

    D,Nx,Ny,Nz=2,2000,2000,2000
    data_random_generator_ = data_random_generator(fun = my_fun,types=["cart","sto","cart"])
    x, fx, y, fy, z, fz =  data_random_generator_.get_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    multi_plot([(x,fx),(z,fz)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
    fz_extrapolated = op.extrapolation(x,fx,x,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
    multi_plot([(x,fx),(x,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
    fz_extrapolated = op.extrapolation(x,fx,y,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
    multi_plot([(y,fy),(y,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
    fz_extrapolated = op.extrapolation(x,fx,z,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
    multi_plot([(x,fx),(z,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")

def test_nablaT_nabla(my_fun,nabla_my_fun,set_kernel):
    D,Nx,Ny,Nz=2,2000,2000,2000
    data_random_generator_ = data_random_generator(fun = my_fun,nabla_fun = nabla_my_fun, types=["cart","cart","cart"])
    x,y,z,fx,fy,fz,nabla_fx,nabla_fz,Nx,Ny,Nz =  data_random_generator_.get_raw_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    f1 = op.nablaT(x,y,z,op.nabla(x,y,z,fx,set_codpy_kernel = set_kernel,rescale = True))
    f2 = op.nablaT_nabla(x,y,fx)
    multi_plot([(x,f1),(x,f2)],plot_trisurf,projection='3d')


def test_withgenerator(my_fun):
    set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel,0,1e-8,map_setters.set_standard_min_map)
    D,Nx,Ny,Nz=2,1000,1000,1000
    scenarios_list = [ (D, 100*i, 100*i ,100*i ) for i in np.arange(1,5,1)]
    if D!=1: projection="3d"
    else: projection=""

    data_random_generator_ = data_random_generator(fun = my_fun,types=["cart","sto","cart"])
    x,y,z,Nx,Ny,Nz =  data_random_generator_.get_raw_data(D=1,Nx=5,Ny=1000,Nz=0)

    x, fx, y, fy, z, fz =  data_random_generator_.get_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    
    multi_plot([(x,fx),(z,fz)],plotD,mp_title="x,f(x)  and z, f(z)",projection=projection)

    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,data_random_generator_,codpyexRegressor(set_kernel = set_kernel),
data_accumulator())
    list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 2)





if __name__ == "__main__":
    def my_fun(x):
        import numpy as np
        from math import pi
        D = len(x)
        res = 1.;
        for d in range(0,D):
            res *= np.cos(2 * x[d] * pi) 
        for d in range(0,D):
            res += x[d]
        return res
    # test_predictor(my_fun)
    # test_withgenerator(my_fun)
    def nabla_my_fun(x):
        import numpy as np
        from math import pi
        D = len(x)
        a = np.zeros((D))
        cost = 1.;
        for d in range(0,D):
            cost *= np.cos(2 * x[d] * pi) 
        for d in range(0,D):
            a[d] = 1
            if cost != 0.:
                a[d] += 2.* cost * pi*np.sin(2* x[d] * pi) / np.cos(2 * x[d] * pi)
            return a

    set_kernel = kernel_setters.kernel_helper(
    kernel_setters.set_tensornorm_kernel, 0,1e-8 ,map_setters.set_unitcube_map)
    test_predictor(my_fun)
    test_nablaT_nabla(my_fun,nabla_my_fun,set_kernel)
