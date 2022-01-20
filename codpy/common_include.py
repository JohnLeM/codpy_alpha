import os,sys
import numpy as np
#######global variables#######
parent_path = os.path.dirname(__file__)
if parent_path not in sys.path: sys.path.append(parent_path)
common_path = os.path.join(parent_path,"common")
if common_path not in sys.path: sys.path.append(common_path)
data_generators_path = os.path.join(parent_path,"data")
if data_generators_path not in sys.path: sys.path.append(data_generators_path)
predictors_path = os.path.join(parent_path,"predictors")
if predictors_path not in sys.path: sys.path.append(predictors_path)
algorithms_path = os.path.join(parent_path,"Algorithms")
if algorithms_path not in sys.path: sys.path.append(algorithms_path)
clustering_path = os.path.join(parent_path,"Clustering")
if clustering_path not in sys.path: sys.path.append(clustering_path)
radon_path = os.path.join(parent_path,"radon")
if radon_path not in sys.path: sys.path.append(radon_path)
time_series_path = os.path.join(parent_path,"TimeSeries")
if time_series_path not in sys.path: sys.path.append(time_series_path)
codpy_book_path = os.path.join(parent_path,"CodPy--book")
codpy_book_fig_path = os.path.join(codpy_book_path,"CodPyFigs")
from codpy_tools import *
plt.rc('font', size=6)
#######################################
