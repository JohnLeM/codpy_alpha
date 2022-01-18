import os,sys
import time as time
import codpy as cd
from codpy_tools import *
from scikit_tools import *
import matplotlib.pyplot as plt
import pylab 
import scipy.stats as stats
import numpy as np

#### global variables
N = 100
D = 1
###

def output(fx,fz, msg = "", plotit = True):
    print("###############",msg,"######################")
    if plotit == True:
        compare_plot(fx,fz)
    print("summary:")
    print(summary((fx,fz)))
    print("#####################################")
    print("KS test:")
    print(ks_testD(fx,fz))
    print("#####################################")
    print("compare_distances:")
    print(compare_distances(fx,fz))
    print("#####################################")


def ecdf(data, idx = 0, reshape=False, fun_permutation = lexicographical_permutation):
    import numpy as np
    if (data.ndim == 2):
        x = data[:,idx]
        x.sort()
    elif (data.ndim == 1):
        x = np.sort(data)
    else:
        raise 'HiThere'
    n = len(x)
    y = np.arange(start = .5/n,stop = 1.,step = 1./n).reshape(n)
    if (reshape): 
        x = x.reshape((len(x),1))
        y = y.reshape((len(y),1))
    return(x,y)

def quantile(fz, idx = 0, reshape=False, fun_permutation = lexicographical_permutation):
    (x,y, permutation) = ecdf(data = fz, idx = idx, reshape=reshape, fun_permutation = lexicographical_permutation)
    return(y,x)

def epsilon(n, alpha=0.05):
   return np.sqrt(1. / (2. * n) * np.log(2. / alpha))


def gendata(type, location = 0, scale = 1, size = (N,D), df=3):
    import scipy
    arg = dist[type]
    switcher = {
        1: np.random.normal(location, scale, size),
        2: np.random.standard_t(df,size)
    }
    return switcher.get(arg, 'nothing')


def plot_ecdf(x,y):
    plt.scatter(x,y)

def QQplot(x,y,fx,fy,idx=0, title = "compare functions"):
    import matplotlib.pyplot as plt
    if x.ndim > 1:
        x = x[x[:,idx].argsort()].reshape(len(x))
    if y.ndim > 1:
        y = y[y[:,idx].argsort()].reshape(len(y))
    if fx.ndim > 1:
        fx = fx[fx[:,idx].argsort()].reshape(len(fx))
    if y.ndim > 1:
        fy = fy[fy[:,idx].argsort()].reshape(len(fy))
    plt.plot(x, fx, marker = 'o',color='red',label='orginal',linewidth=1, markersize=2)
    plt.plot(y, fy, marker = 'o',color='blue',label='extrapolated',linewidth=1, markersize=2)
    pylab.show()

# def compare_plot(fx,fz,title="scatter compare"):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     dim = np.shape(fx)[1]
#     if (dim == 1):cdfs_compare_plot(fx,fz,title)
#     else:
#         plt.scatter(x = fx[:,0],y = fx[:,1],color = 'blue')
#         plt.scatter(x = fz[:,0],y = fz[:,1],color = 'red')
#         plt.show() 

def summary(xs):
    import numpy as np
    import scipy
    import pandas as pd
    df = pd.DataFrame() 

    if (len(xs)):
        dim = np.shape(xs[0])[1]
        columns = {}
        for idx in range(0,dim):
            columns.update({'skew'+str(idx) : [] })
            columns.update({'kurtosis'+str(idx) : [] })
        for x in xs:
            for idx in range(0,dim):
                s = stats.skew(x[:,idx])
                k = stats.kurtosis(x[:,idx])
                columns['skew'+str(idx)].append(s)
                columns['kurtosis'+str(idx)].append(k)
        df = pd.DataFrame(data = columns) 
    return df 

def chi2t_plot():
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 100)
    fig,ax = plt.subplots(1,1)

    linestyles = [':', '--', '-.', '-']
    df = [1, 2, 3, 4]
    for dfi, ls in zip(df, linestyles):
        ax.plot(x, stats.chi2.pdf(x, dfi), linestyle=ls)

    plt.xlim(0, 10)
    plt.ylim(0, 1)

    plt.xlabel("Pearson's cumulative test statistic" )
    plt.ylabel('p-value')
    plt.title('Chi-Square Distribution')
    plt.show()

def chi2test(fx,fz):
    #goodness_of_fit_test
    import scipy
    import pandas as pd
    k = fx.shape[0]
    chisq = (fx-fz) ** 2 / fz
    stat = chisq.sum(axis = 0)
    degf = k - 1
    pvalue = stats.chi2.cdf(stat,df=degf)
    df = {'statistic': stat, 'p-value': pvalue}    
    df = pd.DataFrame(df) 
    return df   

from collections import namedtuple
ks_test_output = namedtuple('ks_test', ('statistic', 'pvalue'))
def ks_test(x, y):
    import scipy
    # print (x.T)
    # print (y.T)
    out = stats.ks_2samp(x, y)
    return ks_test_output(out)
    # Hodges, J. L. (1958). 
    # The significance probability of the Smirnov two-sample test.
    ###########################################################
    # [STE1] Stephens M A (1974) EDF tests for goodness of fit and 
    # some comparisons. J Amer. Statistical Assoc., 69(347), 730-737
    #x = np.sort(x) 
    #y = np.sort(y)
    #s1 = x.size
    #s2 = y.size
    #xy = np.concatenate([x, y])
    #ecdf1 = np.searchsorted(x, xy, side='right') / (1.0 * s1)
    ##x[x <= v].size / x.size
    #ecdf2 = np.searchsorted(y, xy, side='right') / (1.0 * s2)
    #stat = np.max(np.absolute(ecdf1 - ecdf2)) # stat absolute not signed distance
    #ratio = np.sqrt(s1 * s2 / float(s1 + s2)) #eq. 2.5 (H1958)
    #try:
    #    #KS survival function (1-CDF)
    #    pvalue = scipy.stats.kstwobign.sf((ratio + 0.12 + 0.11 / ratio) * stat) 
    #    #STE1974
    #except:
    #    pvalue = 1.0 #prob  
    #return ks_test_output(stat, pvalue)

def ks_testD(x,y):
    import numpy as np
    import scipy
    import pandas as pd
    dim = np.shape(x)[1]
    ks = []
    for i in range(0,dim):
        z = stats.ks_2samp(x[:,i],y[:,i])
        ks.append(z)
    df = pd.DataFrame(ks)  
    return(df)

def hellinger(p,q, idx = 0):
    # p,q are two distribution functions
    import scipy
    import numpy
    if len(p) != len(q): return float('NaN')
    (xp,yp) = ecdf(p)
    (xq,yq) = ecdf(q)
    (xpb,ypb) = (xp.copy(),yp.copy())
    min = numpy.amin((xp,xq))
    numpy.insert(arr = xpb, obj = 0, values = (min))
    min = numpy.amin((yp,yq))
    numpy.insert(arr = ypb, obj = 0, values = (min))
    max = numpy.amax((xp,xq))
    numpy.append(xpb,[max])
    max = numpy.amax((yp,yq))
    numpy.append(ypb,[max])

    fun = scipy.interpolate.interp1d(x=xp,y=yp,axis=0,bounds_error=False,fill_value=(0.,1.))
    from scipy.linalg import norm
    sqrt2 = np.sqrt(2)
    H = np.sqrt(fun(xq))
    H = H-np.sqrt(yq)
    H = norm(H) / sqrt2
    #df = {'H(p;q)': [H]} 
    return H


def codpy_distance(p,q, type, rescale = True):
    if(type == 'H'):
        return hellinger(p,q)
    elif(type == 'D'):
        import codpy as cd
        return op.discrepancy(p,q, rescale = rescale)

def compare_distances(p,q):
    import pandas as pd
    H = codpy_distance(p,q,'H')
    D = codpy_distance(p,q,'D') 
    df = {'Hellingers': [H], 'discrepancy': [D]} 
    df = pd.DataFrame(df)     
    return df   

def cdfs_compare_plot(fx,fz,title="compare cdfs",labelx='fx-axis:',labelz='fz-axis:'):
    import matplotlib.pyplot as plt
    dim = np.shape(fx)
    fig = plt.figure()

    for idx in range(0,dim[1]):
        e1,e2 = ecdf(fx[:,idx])
        e3,e4 = ecdf(fz[:,idx])
        ax=fig.add_subplot(1,dim[1],idx+1)
        ax.plot(e1,e2,marker = 'o',ls='-',label= labelx+str(idx),markersize=2)
        ax.plot(e3,e4,marker = 'o',label=labelz+str(idx),markersize=2)
        ax.set_title(title)
        if (dim[1] == 1):
            plt.xlabel(labelx)
            plt.ylabel(labelz)
        else:
            plt.xlabel(labelx+":"+str(idx))
            plt.xlabel(labelz+":"+str(idx))

    plt.show()           

def compare_plot(fx,fz,title="compare distribution",labelx='fx-axis:',labelz='fz-axis:', max = 9):
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    dim = np.shape(fx)[1]
    count = 0
    max = min(math.factorial(dim-1),max)
    if (dim == 1):cdfs_compare_plot(fx,fz,title,labelx,labelz)
    elif (dim == 2):
        fig = plt.figure(figsize=(max * 7, 7))
        fig.suptitle(title)
        plt.scatter(x = fz[:,0],y = fz[:,1],color = 'red',alpha = .5)
        plt.scatter(x = fx[:,0],y = fx[:,1],color = 'blue',alpha = .5)
        plt.xlabel(labelx)
        plt.ylabel(labelz)
        plt.show()           
    else:
        fig = plt.figure(figsize=(max * 7, 7))
        fig.suptitle(title)
        for i in range(0,dim):
            for j in range(i+1, dim):
                count = count + 1
                if count> max:
                    break
                fig.add_subplot(1, max, count)
                plt.scatter(x = fx[:,i],y = fx[:,j],color = 'blue')
                plt.scatter(x = fz[:,i],y = fz[:,j],color = 'red')
                plt.xlabel(labelx + str(i))
                plt.ylabel(labelz + str(j))
        plt.show()           

def experiment(Num,N,D, M = N):
    import numpy as np
    H = []
    Dis = []
    for i in range(1,Num):
        p  = cdf('norm',gendata('norm',0,1, (N, D)))
        q  = cdf('norm',gendata('norm',0,1, (M, D)))
        h = codpy_distance(p,q,'H')
        d = codpy_distance(p,q,'D') 
        H.append(h)
        Dis.append(d)
    return(np.mean(H), np.mean(Dis))    

def test(N,D, arg = 'min'):
    import numpy as np
    import pandas as pd
    import codpy as cd

    switcher = {
    1: np.min,
    2: np.mean
    }
    funcs = {'min':1, 'mean': 2}
    fun = switcher.get(funcs.get(arg))

    assert(D>2)

    fx = []
    for i in range(0,D+10,10):
        for j in range(200,N+1,200):
            if i == 0:
                x = np.reshape(np.random.normal(0., 1., (j,1)),(j,1))
                z = alg.sampler(x,j)
                xx = fun(ks_testD(x,z)['pvalue'])
                fx.append(xx)
                dis = op.discrepancy(x,z)
                fx.append(dis)
            else:
                x = np.reshape(np.random.normal(0., 1., (j,i)),(j,i))
                z = alg.sampler(x,j)
                xx = fun(ks_testD(x,z)['pvalue'])
                fx.append(xx)
                dis = op.discrepancy(x,z)
                fx.append(dis)

    NN = np.linspace(200,N,N//200)
    Dim = np.repeat(np.linspace(0,D,D//10+1),2)
    if D < 10:
        Dim = np.repeat(np.linspace(0,D,D//2),2)

    Dim[0] = Dim[1] = 1
    print('##################################################################')
    print('KS test / Discrepancy error')
    return(pd.DataFrame(np.reshape(fx,(len(NN),len(Dim))), columns = tuple(Dim), index = tuple((NN))))


if __name__ == "__main__":
    print("OK")
    #N = 1000
    #D = 1
    #print(experiment(1000,N,D))
    #random_sample = np.random.multinomial(100, fx[:,0])
    # N = 500
    # M = 1000
    # D = 2
    N = 1000
    D = 10
    print(test(N,D, 'min'))