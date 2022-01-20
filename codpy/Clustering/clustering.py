import os, sys
import time 
import numpy as np
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:sys.path.append(parentdir)
from common_include import *
from data_generators import *
from predictors import *
from mnist_codpy import *

####A standard run

def cluster_impl(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs):
    scenario_generator.run_scenarios(scenarios_list,generator,predictor,accumulator,**kwargs)
    if bool(kwargs.get("Show_results",True)):
        results = accumulator.get_output_datas().dropna(axis=1)
        print(results)
    if bool(kwargs.get("Show_confusion",True)):accumulator.plot_confusion_matrices(**kwargs,mp_title = "confusion matrices for "+predictor.id())
    if bool(kwargs.get("Show_clusters",True)):accumulator.plot_clusters(**kwargs, mp_title = "PCA clusters for "+predictor.id())
    if bool(kwargs.get("Show_clusters_index",True)):accumulator.plot_clusters(**kwargs, mp_title = "clusters for "+predictor.id(),index1=0,index2=1,xlabel = 'x',ylabel = 'y')
    if bool(kwargs.get("Show_maps_cluster",True)):print(accumulator.get_maps_cluster_indices())


class cluster_accumulator(data_accumulator):
    def set_data(self,generators =[], predictors = [],**kwargs):
        super().set_data(generators =generators, predictors = predictors,**kwargs)
    def accumulate(self,predict,generator,**kwargs):
        super().accumulate(predict,generator,**kwargs)
    def get_score_silhouettes(self):
        return  np.asarray([np.round(s.score_silhouette,2) for s in self.predictors])
    def get_inertias(self):
        return  np.asarray([np.round(s.inertia,2) for s in self.predictors])
    def get_score_calinski_harabaszs(self):
        return  np.asarray([np.round(s.score_calinski_harabasz,2) for s in self.predictors])
    def get_homogeneity_test(self):
        return  np.asarray([np.round(s.homogeneity_test,2) for s in self.predictors])

    def copy_data(self,out):
        super().copy_data(out)
        return out
    def copy(self):
        return self.copy_data(cluster_accumulator())
    def get_output_datas(self):
        super_numbers = super().get_output_datas()
        errors_calinski = self.format_helper(self.get_score_calinski_harabaszs())
        errors_silhouettes = self.format_helper(self.get_score_silhouettes())
        homogeneity_test = self.format_helper(self.get_homogeneity_test())
        inertia = self.format_helper(self.get_inertias())
        numbers = np.concatenate((errors_calinski, errors_silhouettes, homogeneity_test, inertia), axis=1)
        numbers = pd.DataFrame(data=numbers, columns=["scores_calinsky", "score_harabazs", "homogeneity_test", "inertia"])
        numbers = pd.concat((super_numbers,numbers),axis=1)
        # print(numbers)
        return  numbers

########################## a mettre dans des accumulateurs dédiés
    def plot_stock_data(self, engin = True):
        scenario = self.get_numbers()
        d,nx,ny,nz = scenario[0]
        from sklearn.cluster import KMeans
        t = self.get_xs()[0]
        fx = KMeans(n_clusters=ny,
            init='k-means++', 
            n_init=ny, 
            max_iter=300, 
            random_state=0).fit(self.get_zs()[0]).fit_predict(t)
        if engin == True:
            self.df_plot = (self.get_zs()[0]).copy(deep=True)
            self.df_plot["Close"] = t["Close"]
            self.df_plot["Volume"] = t["Volume"]
            self.df_plot["clustering"] = self.get_f_zs()[0]
        else: 
            self.df_plot = t.copy(deep=True)
            self.df_plot["clustering"] = fx
        return self.plot_market_regimes()

    def plot_market_regimes(self):
        import numpy as np
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        scenario = self.get_numbers()
        d,nx,ny,nz = scenario[0] #n_regimes = ny
        cmap = sns.color_palette("muted", ny)
        for n in range(0, ny):
            close = self.df_plot[self.df_plot["clustering"]==n]["Close"]
            index, values = zip(*close.items())
            close = pd.DataFrame({'values': values}, index=pd.DatetimeIndex(index))
            close = close.asfreq('D')
            plt.plot(close, color = cmap[n])
        plt.show()
    
    def plot_stock_clusters(self, all = False):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import itertools
        scenario = self.get_numbers()
        d,nx,ny,nz = scenario[0]
        cmap = sns.color_palette("muted", ny)
        if all == True:
            plt.figure()
            sns.scatterplot(data = self.df_plot, y="Volume", x="Return", hue="clustering", palette = cmap)
            plt.show()
        else:
            fig, axes = plt.subplots(1, ny)
            for i, j in itertools.product(list(range(0,2)), list(range(0,ny))):
                sns.scatterplot(ax=axes[j], y=self.df_plot[self.df_plot["clustering"]==j]["Volume"],
                x=self.df_plot[self.df_plot["clustering"]==j]["Return"], color = cmap[j])
                axes[j].set_title(j)
        plt.show()

##################################### Kernels
set_gaussian_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8,map_setters.set_mean_distance_map)
set_tensornorm_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 0,0,map_setters.set_unitcube_map)
##################################### Blob
    
def blob(**codpy_param):
    set_kernel = set_gaussian_kernel
    scenarios_list = codpy_param.get("scenario_list",[ (2, 1000, i,1000 ) for i in np.arange(2,9,1)])
    validator_compute=['accuracy_score','discrepancy_error','inertia']
    scenario_generator_ = scenario_generator()
    cluster_impl(scenario_generator_,scenarios_list,data_blob_generator(),MinibatchClusterClassifier(set_kernel = set_kernel),cluster_accumulator(), **codpy_param)
    cluster_impl(scenario_generator_,scenarios_list,data_blob_generator(),scikitClusterClassifier(set_kernel = set_kernel),cluster_accumulator(), **codpy_param)
    cluster_impl(scenario_generator_,scenarios_list,data_blob_generator(),codpyClusterClassifier(set_kernel = set_kernel),cluster_accumulator(),**codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Ny","scores"),("Ny","discrepancy_errors"),("Ny","inertia"),("Ny","execution_time")],
        mp_title = "Benchmark methods"
    )
    scenario_generator_.compare_plot(axis_label = "Ny",field_label="discrepancy_errors")
    # ###################################

def mnist_impl(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs):
    cluster_impl(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs)
    multi_plot(accumulator.get_ys(),fun_plot = show_mnist_pictures,mp_ncols = 1, mp_max_items = 10)

 
def mnist(**codpy_param):
    set_kernel = set_gaussian_kernel
    scenarios_list = [ (-1, 6000, 2**i,-1) for i in np.arange(7,9,1)]
    validator_compute=['accuracy_score','discrepancy_error','inertia']
    scenario_generator_ = scenario_generator()
    mnist_impl(scenario_generator_,scenarios_list,MNIST_data_generator(),MinibatchClusterClassifier(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=False,
        **codpy_param)
    mnist_impl(scenario_generator_,scenarios_list,MNIST_data_generator(),codpyClusterClassifier(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=False,
        **codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Ny","scores"),("Ny","discrepancy_errors"),("Ny","inertia"),("Ny","execution_time")],
        mp_title = "Benchmark methods"
    )

def company_stock_movements(**codpy_param):
    set_kernel = set_gaussian_kernel
    scenarios_list = [(-1, -1, i,-1) for i in range(10, 21,10)]
    validator_compute=['discrepancy_error','inertia']
    scenario_generator_ = scenario_generator()

    cluster_impl(scenario_generator_,scenarios_list,company_stock_movements_data_generator(),scikitClusterPredictor(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=True,
        **codpy_param)
    cluster_impl(scenario_generator_,scenarios_list,company_stock_movements_data_generator(),codpyClusterPredictor(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=True,
        **codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Ny","discrepancy_errors"),("Ny","inertia"),("Ny","execution_time")],
        mp_title = "Benchmark methods / Elbow"
    )

def german_credit(**codpy_param):
    set_kernel = set_gaussian_kernel
    scenarios_list = [(-1, -1, i,-1) for i in range(10, 21,10)]
    validator_compute=['discrepancy_error','inertia']
    scenario_generator_ = scenario_generator()
    cluster_impl(scenario_generator_,scenarios_list,german_credit_data_generator(),scikitClusterPredictor(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=True,
        **codpy_param)
    cluster_impl(scenario_generator_,scenarios_list,german_credit_data_generator(),codpyClusterPredictor(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=True,
        **codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Ny","discrepancy_errors"),("Ny","inertia"),("Ny","execution_time")],
        mp_title = "Benchmark german_credit"
    )



def SP500_Yahoo(**codpy_param):
    set_kernel = set_gaussian_kernel
    scenarios_list = [(-1, -1, i,-1) for i in np.arange(2,21,3)]
    validator_compute=['discrepancy_error','inertia']
    scenario_generator_ = scenario_generator()

    cluster_impl(scenario_generator_,scenarios_list,SP500_Yahoo_generator(),scikitClusterPredictor(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=True,
        **codpy_param)
    cluster_impl(scenario_generator_,scenarios_list,SP500_Yahoo_generator(),codpyClusterPredictor(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=True,
        **codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Ny","discrepancy_errors"),("Ny","inertia"),("Ny","execution_time")],
        mp_title = "Benchmark SP500_Yahoo"
    )

def credit_card_data(**codpy_param):
    set_kernel = set_gaussian_kernel
    scenarios_list = [(-1, -1, i,-1) for i in np.arange(2,21,3)]
    scenario_generator_ = scenario_generator()
    validator_compute=['discrepancy_error','inertia']

    cluster_impl(scenario_generator_,scenarios_list,credit_card_data_generator(),scikitClusterPredictor(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=True,
        **codpy_param)
    cluster_impl(scenario_generator_,scenarios_list,credit_card_data_generator(),codpyClusterPredictor(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = False,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=True,
        **codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Ny","discrepancy_errors"),("Ny","inertia"),("Ny","execution_time")],
        mp_title = "Benchmark credit_card_data"
    )

def credit_card_fraud(**codpy_param):
    set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8,map_setters.set_mean_distance_map)
    scenarios_list = [( -1, 500, i,-1 ) for i in np.arange(60,100,15)]
    scenario_generator_ = scenario_generator()

    cluster_impl(scenario_generator_,scenarios_list,credit_card_fraud_data_generator(),scikitClusterClassifier(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = True,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=False,
        mp_max_items = len(scenarios_list),
        **codpy_param)
    cluster_impl(scenario_generator_,scenarios_list,credit_card_fraud_data_generator(),codpyClusterClassifier(set_kernel = set_kernel),cluster_accumulator(),
        Show_confusion = True,Show_clusters=False,Show_clusters_index = False,Show_maps_cluster=False,
        mp_max_items = len(scenarios_list),
        **codpy_param)
    scenario_generator_.compare_plots(
        axis_field_labels = [("Ny","discrepancy_errors"),("Ny","inertia"),("Ny","execution_time")],
        mp_title = "Benchmark credit_card_data"
    )

def main_test(**codpy_param):
    blob(**codpy_param, scenario_list=[ (2, 100, i,100 ) for i in np.arange(5,10,5)])
    mnist(**codpy_param)
    company_stock_movements(**codpy_param)
    german_credit(**codpy_param)
    SP500_Yahoo(**codpy_param)
    credit_card_data(**codpy_param)
    credit_card_fraud(**codpy_param)

codpy_param = {'rescale:xmax': 1000,
'rescale:seed':42,
'sharp_discrepancy:xmax':1000,
'sharp_discrepancy:seed':30,
'sharp_discrepancy:itermax':5,
'discrepancy:xmax':500,
'discrepancy:ymax':500,
'discrepancy:zmax':500,
'discrepancy:nmax':2000,
'num_threads':25,
'validator_compute':['accuracy_score','discrepancy_error','inertia'],
}

def get_params() : return codpy_param

def blob_test():
    blob(**get_params(), scenario_list=[ (2, 100, i,100 ) for i in np.arange(5,10,5)])

if __name__ == "__main__":
    blob(**get_params(), scenario_list=[ (2, 100, i,100 ) for i in np.arange(5,10,5)])
    # main_test(**codpy_param)
    pass
