import os, sys
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.append(parentdir)
data_path = os.path.dirname(__file__)
from common_include import * 
from data_generators import *



####predictors
class scikitClusterPredictor(standard_cluster_predictor,add_confusion_matrix):
    def copy(self):
        return self.copy_data(scikitClusterPredictor())
    def predictor(self,**kwargs):
        from sklearn.cluster import KMeans
        init = kwargs.get('init','k-means++')
        random_state = kwargs.get('random_state',None)
        kmeans = KMeans(n_clusters=self.Ny, init = init, 
        random_state = random_state).fit(self.x)
        self.estimator = kmeans
        self.y = kmeans.cluster_centers_
        self.fy = kmeans.labels_
        self.f_z = kmeans.predict(self.z)
    def id(self,name = ""):
        return "k-means."


class scikitClusterClassifier(standard_cluster_predictor,add_confusion_matrix):
    def copy(self):
        return self.copy_data(scikitClusterClassifier())
    def predictor(self,**kwargs):
        import pandas as pd
        from sklearn.cluster import KMeans
        # import ctypes 
        # mkl_rt = ctypes.WinDLL('mkl_rt.1.dll')
        # mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(10)))

        kmeans = KMeans(n_clusters=self.Ny, random_state=1).fit(self.x)
        self.f_x = kmeans.predict(self.x) 
        self.estimator = kmeans
        self.y = kmeans.cluster_centers_ 
        test = kmeans.predict(self.z)
        self.f_z = remap(test,get_surjective_dictionnary(self.f_x,self.fx))
    def id(self,name = ""):
        return "k-means"


class MinibatchClusterClassifier(scikitClusterClassifier):
    def copy(self):
        return self.copy_data(MinibatchClusterClassifier())
    def predictor(self,**kwargs):
        from sklearn import metrics
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters = self.Ny).fit(self.x)
        self.f_x = kmeans.predict(self.x) 
        self.estimator = kmeans
        self.y = kmeans.cluster_centers_ 
        self.f_z = remap(kmeans.predict(self.z),get_surjective_dictionnary(self.f_x,self.fx))
    def id(self,name = ""):
        return "minibatch"

class scikitDBSCANPredictor(standard_cluster_predictor):
    def copy(self):
        return self.copy_data(scikitDBSCANPredictor())
    def predictor(self,**kwargs):   
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=0.05, min_samples=5)
        self.f_z = model.fit_predict(self.z)
        self.f_y = model.fit(self.z).labels_
    def id(self,name = ""):
        return "DBSCAN"

class scikitSpectralPredictor(standard_cluster_predictor):
    def copy(self):
        return self.copy_data(scikitSpectralPredictor())
    def predictor(self,**kwargs):   
        from sklearn.cluster import SpectralClustering
        model = SpectralClustering(n_clusters=2, 
        affinity='nearest_neighbors', assign_labels='kmeans')
        self.f_z = model.fit_predict(self.z)
    def id(self,name = ""):
        return "spectral"


##################################### Kernels
set_gaussian_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8,map_setters.set_mean_distance_map)
set_tensornorm_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 0,0,map_setters.set_unitcube_map)
set_per_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)
##################################### Blob
    
def random(**codpy_param):
    set_kernel = set_gaussian_kernel
    scenarios_list = [ (1, 1000, i,1000 ) for i in np.arange(2,9,1)]
    validator_compute=['accuracy_score','discrepancy_error','norm_function']
    scenario_generator_ = scenario_generator()
    standard_supervised_run(scenario_generator_,scenarios_list,data_blob_generator(),scikitClusterClassifier(set_kernel = set_per_kernel),data_accumulator(),validator_compute = validator_compute, **codpy_param)
    standard_supervised_run(scenario_generator_,scenarios_list,data_blob_generator(),codpyClusterClassifier(set_kernel = set_per_kernel),data_accumulator(),validator_compute = validator_compute,**codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Ny","scores"),("Ny","discrepancy_errors"),("Ny","inertia"),("Ny","execution_time")],
        mp_title = "Benchmark methods"
    )



    #scenario_generator_.compare_plot(axis_label = "Ny",field_label="discrepancy_errors")
    # ###################################

if __name__ == "__main__":
    print('loaded')
    # set_kernel = set_gaussian_kernel
    # scenarios_list = [ (2, 500, i,500 ) for i in np.arange(2,20,1)]
    # scenario_generator_ = scenario_generator()
    # scenario_generator_.run_scenarios(scenarios_list,data_moon_generator(),MinibatchClusterClassifier(set_kernel = set_kernel),data_accumulator())
    # z = scenario_generator_.accumulator.get_zs()[2]
    # f_z = scenario_generator_.accumulator.get_f_zs()[2]
    # fz = scenario_generator_.accumulator.get_fzs()[2]
    # dp = data_plots()
    # dp.scatter_plot_x_fx(z,fz, "Moons")
    # dp.scatter_plot_x_fx(z,f_z, "Scikit K-means")
    # scenario_generator_.run_scenarios(scenarios_list,data_moon_generator(),scikitSpectralPredictor(set_kernel = set_kernel),data_accumulator())
    # z = scenario_generator_.accumulator.get_zs()[2]
    # f_z = scenario_generator_.accumulator.get_f_zs()[2]
    # #results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
    # dp.scatter_plot_x_fx(z,f_z, "Spectral")

    # scenario_generator_ = scenario_generator()
    # scenario_generator_.run_scenarios(scenarios_list,data_circles_generator(),MinibatchClusterClassifier(set_kernel = set_kernel),data_accumulator())
    # z = scenario_generator_.accumulator.get_zs()[2]
    # f_z = scenario_generator_.accumulator.get_f_zs()[2]
    # fz = scenario_generator_.accumulator.get_fzs()[2]
    # dp = data_plots()
    # dp.scatter_plot_x_fx(z,fz, "Circles")
    # dp.scatter_plot_x_fx(z,f_z, "Scikit K-means")
    # scenario_generator_.run_scenarios(scenarios_list,data_circles_generator(),scikitSpectralPredictor(set_kernel = set_kernel),data_accumulator())
    # z = scenario_generator_.accumulator.get_zs()[2]
    # f_z = scenario_generator_.accumulator.get_f_zs()[2]
    # #results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
    # dp.scatter_plot_x_fx(z,f_z, "Spectral")