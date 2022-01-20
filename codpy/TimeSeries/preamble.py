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
from scikit_tools import * 
plt.rc('font', size=6)
print("preamble ok") 
########################################
