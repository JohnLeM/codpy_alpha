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
import torch



def mean_absolute_percentage_error(y_true, y_pred): 
    out = 100.*np.mean(np.abs((y_true - y_pred) / y_true))
    return out

class ts_data:

    def get_param(**kwargs):
        sep = kwargs.get("sep",";")
        csv_file = kwargs.get('csv_file',None)
        begin_date = kwargs.get('begin_date',None)
        end_date = kwargs.get('end_date',None)
        date_format =       kwargs.get('date_format','%d/%m/%Y')
        csv_date_format =   kwargs.get('csv_date_format','%m/%d/%Y %H:%M')
        time_col = str(kwargs.get('time_col','Date'))
        select_columns = kwargs.get('select_columns',None)
        return sep,csv_file,begin_date,end_date,date_format,csv_date_format,time_col,select_columns

    def date_helper(xs,date_format):
        if isinstance(xs,list): return [ts_data.date_helper(x,date_format) for x in xs]
        return get_float(pd.to_datetime(xs, format=date_format)) 


    def get_yf_ts_data(**kwargs):

        data = ts_data.get_csv_ts_data(**kwargs)
        if data is not None: return data

        import yfinance as yf
        sep,csv_file,begin_date,end_date,date_format,csv_date_format,time_col,select_columns=ts_data.get_param(**kwargs)

        

        symbols = kwargs.get('symbols',AssertionError("get_yf_data: symbols must be input"))
        yf_symbols = " ".join(symbols)
        yf_begin_date = kwargs.get('yf_begin_date','2015-1-1')
        yahoo_columns = kwargs.get('yahoo_columns',None)
        yahoo_date_format = kwargs.get('yahoo_date_format','%Y-%m-%d')


        data = yf.download(symbols,yf_begin_date)
        if yahoo_columns is not None: data = data[yahoo_columns]
        if len(symbols) > 1:
            data = data.droplevel(level=0, axis = 1)

        index = [pd.to_datetime(x, format=csv_date_format).strftime(date_format) for x in data.index] 
        data =  pd.DataFrame(data = data.values, columns = data.columns, index = index)
               
        if begin_date is not None: 
            begin_date = get_float(pd.to_datetime(begin_date, format=date_format))
            data = data.loc[ [ ( ts_data.date_helper(x,csv_date_format) >= begin_date) for x in data.index ] ]
        if end_date is not None: 
            end_date = get_float(pd.to_datetime(end_date, format=date_format))
            data = data.loc[ [ ( ts_data.date_helper(x,csv_date_format) <= end_date) for x in data.index ] ]
        
        if len(symbols) == 1: data.rename(columns = {'Close':symbols[0]}, inplace = True)
        if csv_file is not None: data.to_csv(csv_file,sep = sep,index = True)

        return data
 

    def get_csv_ts_data(**kwargs):

        sep,csv_file,begin_date,end_date,date_format,csv_date_format,time_col,select_columns=ts_data.get_param(**kwargs)
        if csv_file is not None and os.path.exists(csv_file): data = pd.read_csv(csv_file,sep=sep,index_col = 0)
        else: return None

        if begin_date is not None: 
            begin_date = get_float(pd.to_datetime(begin_date, format=date_format))
            data = data.loc[ [ ( ts_data.date_helper(x,csv_date_format) >= begin_date) for x in data.index ] ]
        if end_date is not None: 
            end_date = get_float(pd.to_datetime(end_date, format=date_format))
            data = data.loc[ [ ( ts_data.date_helper(x,csv_date_format) <= end_date) for x in data.index ] ]

        index = [pd.to_datetime(x, format=csv_date_format).strftime(date_format) for x in data.index] 
        data = pd.DataFrame(data = data.values, columns = data.columns, index = index)

        return data

    def interpolate(data,**kwargs):
        kind = str(kwargs.get("kind","linear"))
        bounds_error = bool(kwargs.get('bounds_error',False))
        copy = bool(kwargs.get('copy',False))
        var_col = kwargs.get('var_col',None)
        float_fun = kwargs.get('float_fun',None)

        nulls = [col for col in data.columns if data[col].isnull().sum()]
        for col in nulls:
            fx = data.loc[data[col].notnull()][col].values
            if var_col is None:
                x = data.loc[data[col].notnull()].index.values
                z = data.index.values
            else: 
                x = data.loc[data[col].notnull()][var_col].values
                z = data[var_col].values
            if float_fun is not None: x,z=float_fun(x),float_fun(z)
            data[col] = interpolate(x, fx, z,kind = kind, bounds_error = bounds_error, fill_value= (fx[0],fx[-1]), copy=copy)
            pass 
        return data


def log_return(x,**kwargs):
    log_return_switchDict = { pd.DataFrame: lambda x,**kwargs :  log_return_dataframe(x,**kwargs) }
    def log_return_dataframe(x,**kwargs):
        out = x
        columns = kwargs.get("columns",None)
        if columns is not None:
            cols = get_starting_cols(x.columns, columns)
            out[cols] = log_return_np(out[cols].values,**kwargs)
        else:   
            out = pd.DataFrame(log_return_np(out.values,**kwargs), columns = out.columns, index = x.index)
        return out    
    type_debug = type(x)
    def log_return_np(x,**kwargs):
        x = np.log(x)
        axis = kwargs.get("axis",None)
        if axis is not None:
            out = np.diff(x,axis = axis)
            # out = np.concatenate([ out, np.zeros( [1,out.shape[1]] )])

            if axis==0:debug = x[[0],:]
            else: debug = x[:,[0]] #je sais c'est moche
            out = np.concatenate((debug,out),axis = axis)
        return out

    method = log_return_switchDict.get(type_debug,log_return_np)
    return method(x,**kwargs)



def inv_log_return(x,**kwargs):
    inv_log_return_switchDict = { pd.DataFrame: lambda x,**kwargs :  inv_log_return_dataframe(x,**kwargs) }
    def inv_log_return_dataframe(x,**kwargs):
        out = x
        columns = kwargs.get("columns",None)
        if columns is not None:
            cols = get_starting_cols(x.columns, columns)
            out[cols] = inv_log_return_np(out[cols].values,**kwargs)
        else:   
            out = pd.DataFrame(inv_log_return_np(out.values,**kwargs), columns = out.columns, index = x.index)
        return out    
    type_debug = type(x)
    def inv_log_return_np(x,**kwargs):
        axis = kwargs.get("axis",None)
        if axis is not None:
            out = np.cumsum(x,axis = axis)
            test = out.mean(axis = 0)
            out = np.exp(out)
        return out

    method = inv_log_return_switchDict.get(type_debug,inv_log_return_np)
    return method(x,**kwargs)

class time_serie_generator(data_generator):

    def get_params(**kwargs) :
        return kwargs.get('time_serie_generator',{})

    def get_new_params(self,**kwargs) :
        import copy
        return copy.deepcopy(kwargs)

    def get_date_column_name(self,**kwargs):
        return time_serie_generator.get_params(**kwargs).get("time_id","Date")

    def get_nb_features(self,x,**kwargs):
        return int(x.shape[1] / time_serie_generator.get_P(**kwargs))
    def get_P(**kwargs):
        return int(time_serie_generator.get_params(**kwargs).get("P",1))
    def get_H(**kwargs):
        return int(time_serie_generator.get_params(**kwargs).get("H",1))

    def get_test_size(**kwargs):
        out = time_serie_generator.get_params(**kwargs).get('test_size',None)
        if out is not None: out = float(out)
        return out


    def format_output(self,x, **kwargs):
        if isinstance(x, list) : return [self.format_output(a,**kwargs) for a in x]
        nb_feat = self.get_nb_features(x,**kwargs)
        if nb_feat == 0: nb_feat = x.shape[1]
        step = int(x.shape[1]/nb_feat)
        out = [x.values[[n],:] for n in range(0,x.shape[0],step)]
        out = np.concatenate(out,axis = 1)
        raw_data = self.get_raw_data(**kwargs)
        debug = x.shape[0]%step
        if debug : 
            debug = x.iloc[[-1], -debug*nb_feat:]
            out = np.concatenate([out, debug.values] ,axis = 1)
        output_cols = kwargs.get("output_cols",None)
        if output_cols is not None:
            output_cols0 = [col + "0" for col in output_cols]
        else: 
            output_cols0 = list(x.columns[:nb_feat])

        list_index = [col for col in x.columns if col in output_cols0]
        list_index = [x.columns.get_loc(col) for col in list_index]

        list_values = pd.DataFrame(columns = output_cols0)
        for index,col in zip(list_index,output_cols0):
            list_values[col] = [out[0,n+index] for n in range(0,out.shape[1],nb_feat)]
        out = list_values

        out =  pd.DataFrame(data = out.values, columns = out.columns, index = raw_data.index[-out.shape[0]:])
        return out

    def get_xfx_from_file_or_recompute(format_data_fun,get_data_fun,**kwargs):
        params = time_serie_generator.get_params(**kwargs)
        sep = str(params.get("sep",";"))
        x_raw_data_csv = params.get('raw_data_x_csv',None)
        fx_raw_data_csv = params.get('raw_data_fx_csv',None)
        H = time_serie_generator.get_H(**kwargs)
        P = time_serie_generator.get_P(**kwargs)

        ts_format_ = True
        if (x_raw_data_csv is not None and os.path.exists(x_raw_data_csv)) and (fx_raw_data_csv is not None and os.path.exists(fx_raw_data_csv) ): 
            x,fx = pd.read_csv(x_raw_data_csv,sep=sep, index_col = 0), pd.read_csv(fx_raw_data_csv,sep=sep, index_col = 0)
            if x.shape[1] / fx.shape[1] != H/P : os.remove(x_raw_data_csv),os.remove(fx_raw_data_csv)
            else: ts_format_ = False
        if  ts_format_:   
            x,fx = format_data_fun(x = get_data_fun(**kwargs),h=H,p=P,**kwargs)

        if x_raw_data_csv is not None and not os.path.exists(x_raw_data_csv): x.to_csv(x_raw_data_csv,sep = sep, index = True)
        if fx_raw_data_csv is not None and not os.path.exists(fx_raw_data_csv): fx.to_csv(fx_raw_data_csv,sep = sep, index = True)
        return x,fx
    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):

        from sklearn.model_selection import train_test_split
        x,fx = time_serie_generator.get_xfx_from_file_or_recompute(format_data_fun = ts_format,get_data_fun = self.get_raw_data,**kwargs)
        test_size = time_serie_generator.get_test_size(**kwargs)
        P = time_serie_generator.get_P(**kwargs)

        if test_size is not None: 
            z,fz = x,fx
            x,fx = x.iloc[:int(x.shape[0]*(1.-test_size)),:],fx.iloc[:int(fx.shape[0]*(1-test_size)),:]
        else: 
            x,fx,z,fz = x,fx,x,fx
        return x,fx,x,fx,z,fz

    @abc.abstractmethod
    def get_raw_data(self,**kwargs):
        pass
    def id(self,name = ""):
        return "ts gen"


class time_serie_optimal_generator(time_serie_generator):

    def format_output(self,x, **kwargs):
        return x

    def op_format(x,**kwargs):
        # P = time_serie_generator.get_P(**kwargs)
        # x,fx = x.iloc[:-P,:],x.iloc[P:,:]
        # x,fx = x.iloc[:-1,:],x.iloc[1:,:]
        # return x,fx
        return x,x

    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):
        from sklearn.model_selection import train_test_split
        x,fx = time_serie_generator.get_xfx_from_file_or_recompute(format_data_fun = time_serie_optimal_generator.op_format,get_data_fun = self.get_raw_data,**kwargs)
        test_size = time_serie_generator.get_test_size(**kwargs)
        P = time_serie_generator.get_P(**kwargs)
        H = time_serie_generator.get_H(**kwargs)

        if test_size is not None: 
            z = x.copy()
            fz = fx.copy()
            x,fx = x.iloc[:int(x.shape[0]*(1.-test_size)),:],fx.iloc[:int(fx.shape[0]*(1.-test_size)),:]
        else: 
            x,fx,z,fz = x,fx,x,fx
        # fx,x,permutationfx = alg.reordering(fx,x,set_codpy_kernel = None, rescale = True)
        return x,fx,x,fx,z,fz

    def id(self,name = ""):
        return "ts opt gen"

class forecast_predictor(data_predictor):

    @abc.abstractmethod
    def one_step_predictor(self,x,y,z,fx,**kwargs): pass

    def get_z_fz(self,**kwargs):
        return self.x,self.fx

    def get_params(**kwargs) :
        return kwargs.get('forecast_predictor',{})

    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            # self.f_z = self.fz
            # return
            params = time_serie_generator.get_params(**kwargs)

            H = time_serie_generator.get_H(**kwargs)
            P = time_serie_generator.get_P(**kwargs)
            z,fz = self.get_z_fz(**kwargs)
            # z,fz = self.x,self.fx
            self.set_kernel()
            kernel.rescale(self.x.values,self.x.values,self.z.values,**kwargs)

            while fz.shape[0] < self.fz.shape[0]:
                nb = min(self.fz.shape[0] - fz.shape[0],P)
                # zvalues =z.iloc[-nb:,:]
                # fzvalues = self.one_step_predictor(self.x,self.y,zvalues, self.fx,**kwargs)
                zvalues =z.iloc[-nb:,:].copy()
                fzvalues = self.one_step_predictor(z,z,zvalues, fz,**kwargs)


                fzvalues.index = self.fz.index[fz.shape[0]:fz.shape[0]+fzvalues.shape[0]]
                cols = get_starting_cols(fzvalues.columns, ["Date"])
                if len(cols):
                    fzvalues[cols] = self.fz[cols].iloc[[x in fzvalues.index for x in self.fz.index]]
                    zvalues[cols] = self.z[cols].iloc[[x in zvalues.index for x in self.z.index]]

                if self.z.shape[0] > z.shape[0]+zvalues.shape[0]:
                    zvalues.index = self.z.index[z.shape[0]:z.shape[0]+zvalues.shape[0]]
                if zvalues.shape[1] > fzvalues.shape[1]:
                    debug = [ zvalues.iloc[:,(fzvalues.shape[1]-zvalues.shape[1]):].values, fzvalues.values] 
                    debug = np.concatenate(debug, axis = 1)
                    zvalues = pd.DataFrame(debug,columns = zvalues.columns,index = zvalues.index)
                else : 
                    debug = fzvalues.iloc[:,-zvalues.shape[1]:]
                    zvalues = pd.DataFrame(debug.values,columns = zvalues.columns, index = debug.index)
                z = pd.concat([z,zvalues],axis=0)
                fz = pd.concat([fz,fzvalues],axis=0)

            self.f_z = self.post(fz,**kwargs)
    def post(self,fz,**kwargs): return fz

    def id(self,name = ""):
        return "codpy forecast"


class forecast_predictor_codpy(forecast_predictor):

    def one_step_predictor(self,x,y,z,fx,**kwargs):
        rescale = forecast_predictor_codpy.get_rescale(**kwargs)
        return op.projection(x = x,y = y,z = z, fx = fx,set_codpy_kernel=None,rescale = rescale,**kwargs)


    def get_params(**kwargs) :
        return kwargs.get('forecast_predictor_codpy',{})

    def get_rescale(**kwargs):
        return bool(forecast_predictor_codpy.get_params(**kwargs).get("rescale",True))

    def id(self,name = ""):
        return "codpy forecast"


class forecast_optimal_codpy(forecast_predictor_codpy):
    def get_params(**kwargs) :
        return kwargs.get('forecast_optimal_codpy',{})

    def get_z(self,z,**kwargs):
        cols = get_starting_cols(z.columns, ["Date"])
        cols = [x for x in z.columns if x not in cols]
        debug = forecast_optimal_codpy.get_params(**kwargs) 
        seed = debug.get("seed",42)
        np.random.seed(seed)
        z[cols] = np.random.rand(z.shape[0],len(cols))
        # for col in cols:
        #     z[col] = np.random.choice(z[col],size = z.shape[0])
        return z
     
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            # self.f_z = self.fz
            # return
            self.set_kernel()
            test_size = time_serie_generator.get_test_size(**kwargs)


            z = alg.grid_projection(self.z)
            x = z[:int(z.shape[0]*(1.-test_size))]
            fx = self.fz.iloc[1:x.shape[0]+1,:]
            x,fx,permutation = alg.reordering(x,fx,rescale=True)
            z = self.get_z(z[int(z.shape[0]*(1.-test_size)):],**kwargs)
            f_z = self.one_step_predictor(x,x,z, fx,**kwargs)
            f_z = pd.concat([fx,f_z])

            f_z.index = self.fz.index
            cols = get_starting_cols(f_z.columns, ["Date"])
            if len(cols):
                f_z[cols] = self.fz[cols].iloc[[x in f_z.index for x in self.fz.index]]

            self.f_z = self.post(f_z,**kwargs)

    def id(self,name = ""):
        return "codpy optimal"

class forecast_mean_codpy(forecast_optimal_codpy):

    def get_z(self,z,**kwargs):
        cols = get_starting_cols(z.columns, ["Date"])
        cols = [x for x in z.columns if x not in cols]
        test = z[cols].mean(axis=0)
        z[cols] = 0.5
        return z

class ts_scenario_generator(scenario_generator):
    def run_scenarios(self,list_scenarios,accumulator):
        for scenario in list_scenarios:
            d,nx,ny,nz = scenario.get("numbers",(-1,10000,1000,10000))
            data_generator,predictor = scenario.get("generator",None),scenario.get("predictor",None)
            self.data_generator,self.predictor,self.accumulator = data_generator,predictor,accumulator
            data_generator.set_data(d,nx,ny,nz,**scenario)
            predictor.set_data(**scenario)
            # print("predictor:",self.predictor.id()," d:", d," nx:",nx," ny:",ny," nz:",nz)
            accumulator.accumulate(**scenario)
        if not len(self.results): self.results = accumulator.get_output_datas()
        else: self.results = pd.concat((self.results,accumulator.get_output_datas()))


    def plot_output(self,**kwargs):
        results = [{"fz":generator.format_output(predictor.fz,**kwargs),"f_z":generator.format_output(predictor.f_z,**kwargs),**kwargs} for predictor,generator in zip(self.accumulator.predictors,self.accumulator.generators)]
        results = [{"fz":generator.format_output(predictor.fz,**kwargs),"f_z":generator.format_output(predictor.f_z,**kwargs),**kwargs} for predictor,generator in zip(self.accumulator.predictors,self.accumulator.generators)]
        listxs, listfxs = {},{}
        fixed_columns =  kwargs.get('plot_columns',list(results[-1]["fz"].columns))
        for result in results :
            fz,f_z = result["fz"],result["f_z"]
            plot_columns =  kwargs.get('plot_columns',list(fz.columns))
            xs = [list(fz.index),list(f_z.index)]

            for fixed_col,col in zip(fixed_columns,plot_columns):
                if fixed_col not in listxs.keys():listxs[fixed_col]=[]
                if fixed_col not in listfxs.keys():listfxs[fixed_col]=[]
                fzcol,f_zcol = list(fz[col]),list(f_z[col])
                [listxs[fixed_col].append(x) for x in xs]
                listfxs[fixed_col].append(list(fz[col].values)), listfxs[fixed_col].append(list(f_z[col].values))

        for key in listxs.keys():
            xs,fxs = listxs[key], listfxs[key]
            compare_plot_lists(xs, fxs, ax = None, labely = key, **kwargs)


    def plot_output(self,**kwargs):
        listxs, listfxs = {},{}
        results = kwargs.get('results',[])
        for result in results :
            fz = result
            plot_columns =  kwargs.get('plot_columns',list(fz.columns))

            for col in plot_columns:
                if col not in listxs.keys():listxs[col]=[]
                if col not in listfxs.keys():listfxs[col]=[]
                listxs[col].append(list(fz.index))
                listfxs[col].append(list(fz[col].values))

        for key in listxs.keys():
            xs,fxs = listxs[key], listfxs[key]
            compare_plot_lists(xs, fxs, ax = None, labely = key, **kwargs)

def get_scenarios(**kwargs):
    print("predict")
    ts_generator_ = kwargs.get("my_generator")
    ts_predictor_ = kwargs.get("my_predictor")
    ts_generator_ = [generator(**kwargs,set_data = False) for generator in ts_generator_]
    ts_predictor_ = [predictor(**kwargs,set_data = False) for predictor in ts_predictor_]
    numbers = kwargs.get("numbers",(-1,-1,-1,-1))
    dic_kwargs = [{ "numbers":numbers, "generator":generator,"predictor":predictor, **predictor.get_new_params(**kwargs)} for generator,predictor in zip(ts_generator_,ts_predictor_)]
    scenarios = ts_scenario_generator()
    scenarios.run_scenarios(dic_kwargs, data_accumulator())
    return scenarios