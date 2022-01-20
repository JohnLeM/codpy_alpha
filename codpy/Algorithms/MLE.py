import os, sys
import codpy.codpy as cd
from pathlib import Path
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = Path(dir_path).parent
sys.path.append(os.path.join(dir_path,"common"))
from codpy_tools import *
from stat_tools import *
from scikit_tools import *


def testMLE(set_codpy_kernel = alg.set_sampler_kernel):
    df = 2
    D = 2
    N = 1000
    x = np.random.standard_t(df,size = (N, D))
    y = alg.sampler(x,N)
    output(x,y,msg=" Gaussian 2D ")
    compare_plot(x,y,title=" student 40D ", max=3)
    summary_ = summary((x,y))
    ks_ = ks_testD(x,y)
    dd = compare_distances(x,y)



def xz_gen(N, M, D, type = "sto", index:int = 0):
    import itertools as it
    import numpy as np

    M = (int(M**(1/D)) ) ** D
    N = (int(N**(1/D)) ) ** D

    #print('data_genMD.sizez:',sizez)
    #print('data_genMD.sizex:',sizex)
    nM = int(M**(1/D))
    deltaz = .5/ float(nM)
    nN = int(N**(1/D))
    deltax = .5/ float(nN)
    z = np.linspace(deltaz,1.-deltaz, nM)
    v = z
    for i in range(0, D-1):
        v= tuple(it.product(z,v))
    z = np.reshape(v,(M,D))  
    if type == "sto":
        x = np.random.uniform(0., 1., size = (N,D)) 
    else:
        x = np.linspace(deltax,1.-deltax, nN)
        v = x
        for i in range(0, D-1):
            v= tuple(it.product(x,v))
        x = np.reshape(v,(N,D))  

    x = x[x[:,index].argsort()]
    z = z[z[:,index].argsort()]
    return(np.asarray(x),np.asarray(z),N,M,D)


def test(set_codpy_kernel = alg.set_sampler_kernel):
    (N,D,M) = (100,2,100)
    x = np.random.normal(-5., 1., (int(N/2),D))
    x = np.concatenate( (x,np.random.normal(+5., 1., (int(N/2),D))) )    
    y = alg.sampler(x,M,set_codpy_kernel =set_codpy_kernel, rescale = True)
    #fz = alg.reordering(fx,fz)
    output(x,y,msg=" Gaussian D="+str(D))


def test2(set_codpy_kernel = alg.set_sampler_kernel):
    from matplotlib import pyplot as plt
    set_codpy_kernel()
    D,N,M=1,100,1000
    (z,x,M,N,D) = xz_gen(N=M,M=N,D=D,type = "sto")
    print(N,M,D)
    fx = np.random.normal(-5., 1., (int(N/2),D))
    fx = np.concatenate( (fx,np.random.normal(+5., 1., (int(N/2),D))) )   
    fx = np.asarray(sorted(fx)).reshape((N,D))

    fx,x,permutation = alg.reordering(fx,x,set_codpy_kernel = None, rescale = True)
    z = np.asarray(sorted(z)).reshape((M,D))
    fx_reshaped = fx.reshape((D,N,1))

    print (x)
    print (z)
    #print (z.shape)

    plt.title('x, fx')
    plt.plot(x,fx, marker = 'o',linewidth=2, markersize=4)
    plt.show()

    kernel.rescale(x,z)
    hx = op.nabla_inv(x=x,y=x,z=x,fz=fx_reshaped,set_codpy_kernel = None, rescale = False)
    plt.title('x, hx')
    plt.plot(x,hx, marker = 'o',linewidth=2, markersize=4)
    plt.show()

    # nabla_nabla_inv = op.nablaT_nabla_inv(x=x,y=x,z=x,set_codpy_kernel = None, rescale = False)
    # plt.title('x, nabla_nabla_inv')
    # for n in range(0, N-1,int(N/5)):
    #     plt.plot(x,nabla_nabla_inv[n], marker = 'o',linewidth=2, markersize=4)

    # plt.show()


    #hx = op.nablaT_nabla(x,x,x,hx,set_codpy_kernel = None, rescale = False)
    #plt.title('x, hx')
    #plt.plot(x,hx, marker = 'o',linewidth=2, markersize=4)
    #plt.show()

    deltahx = op.nabla(x,x,x,hx,set_codpy_kernel = None, rescale = False).reshape((N,D))
    #deltahx = np.asarray(sorted(deltahx)).reshape((N,D))
    print (deltahx.T)
    plt.title('x, deltahx')
    plt.plot(x,deltahx, marker = 'o',linewidth=2, markersize=4)
    plt.show()

    deltahz = op.nabla(x,x,z,hx,set_codpy_kernel = None, rescale = False).reshape((M,D))
    #deltahz = np.asarray(sorted(deltahz)).reshape((M,D))
    print (deltahz.T)
    plt.title('z, deltahz')
    plt.plot(z,deltahz, marker = 'o',linewidth=2, markersize=4)
    plt.show()

    chix = fx-deltahx
    plt.title('x, chix')
    plt.plot(x,chix, marker = 'o',linewidth=2, markersize=4)
    plt.show()

    chiz = op.projection(x,x,z,chix,set_codpy_kernel = None, rescale = False)
    plt.title('z, chiz')
    plt.plot(z,chiz, marker = 'o',linewidth=2, markersize=4)
    plt.show()

    fz = chiz + deltahz
    plt.title('z, fz')
    plt.plot(z,fz, marker = 'o',linewidth=2, markersize=4)
    plt.show()


def test3(set_codpy_kernel = alg.set_sampler_kernel):
    D,N,M=1,100,100
    (z,x,M,N,D) = xz_gen(N=M,M=N,D=D,type = "sto")
    print(N,M,D) 
    x_reshaped = x.reshape((D,N,1))
    z_reshaped = z.reshape((D,M,1))
    #print (x.T)
    #print (z.T)
    #print (z.shape)
    #print (z_reshaped.shape)
    #print (x_reshaped)
    from matplotlib import pyplot as plt
    xz = op.projection(x,x,z,x,set_codpy_kernel = set_codpy_kernel, rescale = True)
    plt.title('z, x(z)')
    plt.plot(z,xz, marker = 'o',linewidth=2, markersize=4)
    plt.show()
    coeffsx = op.coefficients(x,x,x,set_codpy_kernel = None, rescale = False)
    plt.title('x,coeffsx')
    plt.plot(x,coeffsx, marker = 'o',linewidth=2, markersize=4)
    plt.show()
    nabla_hx = op.nabla(x,x,x,x,set_codpy_kernel = None, rescale = False).reshape((N,D))
    plt.title('x,nabla x')
    plt.plot(x,nabla_hx, marker = 'o',linewidth=2, markersize=4)
    plt.show()
    nablaT = op.nablaT(x,x,x,x_reshaped,set_codpy_kernel = None, rescale = False)
    plt.title('x,nablaT x')
    plt.plot(x,nablaT, marker = 'o',linewidth=2, markersize=4)
    plt.show()
    hx = op.nablaT_nabla_inv(x,x,x,nablaT,set_codpy_kernel = set_codpy_kernel, rescale = False)
    plt.title('x,nablaT_nabla_inv nablaT x')
    plt.plot(x,nablaT, marker = 'o',linewidth=2, markersize=4)
    plt.show()
    hx = op.nabla_inv(x,x,x,x_reshaped,set_codpy_kernel = set_codpy_kernel, rescale = False)
    plt.title('x,nabla_inv x')
    print (hx.shape)
    print (hx.T)
    plt.plot(x,hx, marker = 'o',linewidth=2, markersize=4)
    plt.show()
    nabla_hx = op.nabla(x,x,x ,hx,set_codpy_kernel = None, rescale = False).reshape((N,D))
    compare_plot1D(x,x,x,nabla_hx,title = 'x,x  x,nabla x')
    nabla_hx = op.Leray_T(x,x,x,x_reshaped,set_codpy_kernel = None, rescale = False).reshape((N,D))
    compare_plot1D(x,x,x,nabla_hx,title = 'x,x  x,Leray_T x')
    plt.show()
    
if __name__ == "__main__":
    #testMLE()
    # set_codpy_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 3,1e-8 ,map_setters.set_standard_min_map)
    set_codpy_kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_norm_kernel, 0,0 ,map_setters.set_standard_mean_map)
    test(set_codpy_kernel = set_codpy_kernel)
    # test2(set_codpy_kernel = set_codpy_kernel)
    #test3(set_codpy_kernel)
    pass




