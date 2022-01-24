from preamble import * 
from time_series import * 

class BTC_time_serie_generator(time_serie_generator):

    def format_output(self,x, **kwargs):
        out = super().format_output(x,**kwargs)
        # return out
        raw_data = self.get_raw_data(**kwargs)
        out.iloc[0,:] = raw_data[raw_data.index == out.index[0]]
        columns = ["BTC-USD","hash-rate"]
        return inv_log_return(out,columns = columns, axis = 0)

    def get_params(**kwargs):
        return kwargs.get('BTC_time_serie_generator',{})
    def get_raw_data(self,**kwargs):
        data = BTC_time_serie_generator.get_raw_data_csv_file(**kwargs)
        if data is None: data = BTC_time_serie_generator.get_BTC_data(**kwargs)
        params = BTC_time_serie_generator.get_params(**kwargs)
        data = log_return(data,axis = 0,columns = ["BTC-USD","hash-rate"])
        date_format = params.get('date_format','%d/%m/%Y')
        data['Date'] = [ts_data.date_helper(x,date_format) for x in data.index]
        return raw_data_column_selector(data,**kwargs)

    def get_raw_data_csv_file(**kwargs):
        params = BTC_time_serie_generator.get_params(**kwargs)
        csv_file = params.get('raw_data_csv',None)
        sep = params.get('sep',';')
        if csv_file is not None and os.path.exists(csv_file): 
            data = pd.read_csv(csv_file,sep=sep,index_col = 0)
            return data


    def get_BTC_data(**kwargs):
        params = kwargs.get('BTC_time_serie_generator',None)
        yf_params = params.get('yf_param',AssertionError("BTC_time_serie_generator.get_raw_data needs yf_param"))
        hr_params = params.get('hr_param',AssertionError("BTC_time_serie_generator.get_raw_data needs hr_param"))
        data = ts_data.get_yf_ts_data(**yf_params)
        hr = ts_data.get_csv_ts_data(**hr_params)

        sep,csv_file,begin_date,end_date,date_format,hr_csv_date_format,time_col,select_columns = ts_data.get_param(**hr_params)

        data = data.merge(hr,how='left',left_index=True, right_index=True)
        data = ts_data.interpolate(data,**params, float_fun = lambda x: ts_data.date_helper(x,date_format))

        raw_data_csv = params.get('raw_data_csv',None)
        if raw_data_csv is not None and not os.path.exists(raw_data_csv): data.to_csv(raw_data_csv,sep = sep, index = True)
        return data

class BTC_ts_mean_generator(BTC_time_serie_generator,time_serie_optimal_generator):
    pass
    
class BTC_ts_optimal_generator(BTC_time_serie_generator,time_serie_optimal_generator):
    pass



class BTC_forecast_predictor_codpy(forecast_predictor_codpy):
    pass

class BTC_mean_predictor_codpy(forecast_mean_codpy):
     def get_new_params(self,**kwargs) :
        import copy
        out = copy.deepcopy(kwargs)
        out['time_serie_generator']['raw_data_x_csv'] = out['time_serie_generator']['raw_data_x_csv'].replace("ts_x","mean_x")
        out['time_serie_generator']['raw_data_fx_csv'] = out['time_serie_generator']['raw_data_fx_csv'].replace("ts_fx","mean_fx")
        return out



class BTC_forecast_optimal_codpy(forecast_optimal_codpy):

    def get_new_params(self,**kwargs) :
        import copy
        out = copy.deepcopy(kwargs)
        out['forecast_optimal_codpy'] = {"seed":np.random.random_integers(100000)}
        out['time_serie_generator']['raw_data_x_csv'] = out['time_serie_generator']['raw_data_x_csv'].replace("ts_x","opt_x")
        out['time_serie_generator']['raw_data_fx_csv'] = out['time_serie_generator']['raw_data_fx_csv'].replace("ts_fx","opt_fx")
        return out


   
global_param = {
    'begin_date':'01/01/2015',
    'end_date':'01/01/2022',
    'yahoo_columns': ['Close'],
    'yf_begin_date': '2015-01-01',
    'H' : 360,
    'P' : 360,
    'symbols' : ["BTC-USD"]
}

# global_param = {
#     'begin_date':'01/01/2020',
#     'end_date':'01/01/2021',
#     'yahoo_columns': ['Date','Close'],
#     'yf_begin_date': '2020-01-01',
#     'yahoo_columns': ['Close'],
#     'H' : 10,
#     'P' : 10,
#     'symbols' : ["BTC-USD"]
# }


params = {
    'rescale:xmax': 1000,
    'rescale:seed':42,
    'sharp_discrepancy:xmax':1000,
    'sharp_discrepancy:seed':30,
    'sharp_discrepancy:itermax':10,
    'discrepancy:xmax':500,
    'discrepancy:ymax':500,
    'discrepancy:zmax':500,
    'discrepancy:nmax':2000,
    'validator_compute':['accuracy_score'],
    # 'set_kernel' : kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, 2,0 ,map_setters.set_unitcube_map),
    # 'set_kernel' : kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,0 ,map_setters.set_grid_map),
    # 'set_kernel' : kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,0 ,map_setters.set(["scale_to_unitcube","log"])),
    'set_kernel' : kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,0 ,map_setters.set_unitcube_map),
    # 'set_kernel' : kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2,0 ,map_setters.set_standard_min_map),
    'BTC_time_serie_generator' : {
        'yf_param' : {
            'symbols':global_param['symbols'],
            'begin_date':global_param['begin_date'],
            'end_date':global_param['end_date'],
            'yahoo_columns': global_param['yahoo_columns'],
            'yf_begin_date': global_param['yf_begin_date'],
            'csv_file' : os.path.join(data_path,"BTC",'-'.join(global_param['symbols'])+'-'+global_param['begin_date'].replace('/','-')+"-"+global_param['end_date'].replace('/','-')+".csv"),
            'date_format' : '%d/%m/%Y',
            'yahoo_date_format':'%Y-m%-d%',            
            'csv_date_format':'%d/%m/%Y'            
        },        
        'hr_param' : {
            'begin_date':global_param['begin_date'],
            'end_date':global_param['end_date'],
            'csv_file' : os.path.join(data_path,"BTC","hash-rate.csv"),
            'date_format' : '%d/%m/%Y',
            'csv_date_format' : '%d/%m/%Y %H:%M',
            'select_columns' : None
        },  
        'date_format' : '%d/%m/%Y',
        'raw_data_csv' : os.path.join(data_path,"BTC","time_serie_generator-"+'-'.join(global_param['symbols'])+'-'+global_param['begin_date'].replace('/','-')+"-"+global_param['end_date'].replace('/','-')+".csv"),
    },
    'data_generator' : {
        # 'variables_cols_drop' : ['BTC-USD','hash-rate'],
        # 'values_cols_drop' : ['BTC-USD','hash-rate']
    },
    'data_predictor' : {
        # 'variables_cols_drop' : ['Date'],
        # 'values_cols_drop' : ['Date']
    },
    'time_serie_generator' : {
        'raw_data_x_csv' : os.path.join(data_path,"BTC","ts_x-"+'-'.join(global_param['symbols'])+'-'+global_param['begin_date'].replace('/','-')+"-"+global_param['end_date'].replace('/','-')+"-H"+str(global_param['H'])+"-P"+str(global_param['P'])+".csv"),
        'raw_data_fx_csv' : os.path.join(data_path,"BTC","ts_fx-"+'-'.join(global_param['symbols'])+'-'+global_param['begin_date'].replace('/','-')+"-"+global_param['end_date'].replace('/','-')+"-H"+str(global_param['H'])+"-P"+str(global_param['P'])+".csv"),
        'H':global_param['H'],'P':global_param['P'],'test_size' : 0.1
        }
}

def get_param(hist_depth=0,pred_depth=0):
    return params


def main():
    # scenarios = get_scenarios(**get_param(),my_generator = [BTC_time_serie_generator], my_predictor = [BTC_forecast_predictor_codpy])
    # scenarios = get_scenarios(**get_param(),my_generator = [BTC_ts_optimal_generator], my_predictor = [BTC_forecast_optimal_codpy])
    # scenarios = get_scenarios(**get_param(),my_generator = [BTC_ts_mean_generator], my_predictor = [BTC_mean_predictor_codpy])
    scenarios = get_scenarios(**get_param(),my_generator = [BTC_ts_mean_generator] + [BTC_ts_optimal_generator for n in range(10)], my_predictor = [BTC_mean_predictor_codpy]+[BTC_forecast_optimal_codpy for n in range(10)])
    results = [scenarios.data_generator.format_output(scenarios.predictor.fz,**get_param())]
    for predictor,generator in zip(scenarios.accumulator.predictors,scenarios.accumulator.generators):
        debug = generator.format_output(predictor.f_z,**get_param()) 
        results.append(debug)
    alphas = [1.,1.]+[(len(results)-i)/(2*len(results)) for i in range(len(results)-1)]
    listlabels = ["observed","mean"]+["generated "+str(i) for i in range(len(scenarios.accumulator.predictors)) ]
    scenarios.plot_output(results = results,listlabels=listlabels,alphas = alphas,fun_x = lambda x : pd.to_datetime(x,format='%d/%m/%Y'),**get_param())


if __name__ == "__main__":
  
    main()
    pass