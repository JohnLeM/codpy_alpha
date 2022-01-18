import os, sys
import time 
import numpy as np
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:sys.path.append(parentdir)
from common_include import * 

def reorderingtest(Nx=10,Ny=6,D=2,set_codpy_kernel = kernel_setters.set_gaussian_kernel, rescale = True, seed = 42):
    import random
    print("run test() to start")
    #map_setters.set_min_distance_map()
    if (seed): np.random.seed(seed)
    left = np.random.normal(-3.,1., (int(Nx/2),D))
    right = np.random.normal(3., 1., (int(Nx/2),D))
    x0 = np.concatenate( (left,right) )
    permutation = [i for i in range(0,len(x0))]
    random.shuffle(permutation)
    x0 = x0[permutation]
    y0 = np.random.rand(Ny,D)

    x,y,permutation = alg.reordering(x0,y0, distance ='norm2' )
    reordering_plot(x0,y0,x,y)
    print(x0,y0)
    print(x,y)

    x,y,permutation = alg.reordering(x0,y0, set_codpy_kernel = set_codpy_kernel, rescale = rescale)
    reordering_plot(x0,y0,x,y)

if __name__ == "__main__":
    reorderingtest()    