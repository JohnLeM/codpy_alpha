import os, sys
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.append(parentdir)
data_path = os.path.dirname(__file__)
from common_include import * 
from data_generators import *
import torch.nn as nn
import torch


class standard_xgboost_predictor(data_predictor, add_confusion_matrix):

    def copy(self):
        return self.copy_data(standard_xgboost_predictor())

    def get_model(self, **kwargs):
        import xgboost as xgb 
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        max_depth = kwargs.get('ax_depth', 6)
        subsample = kwargs.get('subsample', 1)
        learning_rate = kwargs.get('learning_rate', 3)
        colsample_bytree = kwargs.get('colsample_bytree', 1)
        reg_alpha = kwargs.get('reg_alpha', 3)
        n_estimators = kwargs.get('n_estimators', 3)


        return xgb.XGBRegressor(n_estimators = n_estimators, max_depth = max_depth,
        colsample_bytree = colsample_bytree, reg_alpha = reg_alpha).fit(self.x,self.fx)

    def predictor(self,**kwargs):
        model = self.get_model()
        self.f_z = model.predict(self.z)

    def id(self,name = "XGboost"):
        return "XGboost"

class label_xgboost_predictor(data_predictor, add_confusion_matrix):

    def copy(self):
        return self.copy_data(label_xgboost_predictor())

    def get_model(self, **kwargs):
        import xgboost as xgb 
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        epochs = kwargs.get('epochs',5)
        num_class = kwargs.get('num_class', 10)
        objective = kwargs.get('objective','multi:softmax')
        eta = kwargs.get('eta', 3)
        max_depth = kwargs.get('max_depth', 3)

        xgb_param = {'max_depth': max_depth,
        'eta' : eta,
        'objective': objective,
        'num_class': num_class}

        self.train = xgb.DMatrix(self.x, label=self.fx)
        self.test = xgb.DMatrix(self.z, label=self.fz)

        return xgb.train(xgb_param, self.train, epochs)

    def predictor(self,**kwargs):
        model = self.get_model()
        self.f_z = model.predict(self.test)

    def id(self,name = "XGboost"):
        return "XGboost"

#####################################################################
def objective(space):
    import xgboost as xgb 
    from sklearn.metrics import mean_squared_error
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
    import numpy as np
    model = xgb.XGBRegressor(
    max_depth = int(space['max_depth']),
    n_estimators = int(space['n_estimators']),
    subsample = space['subsample'],
    colsample_bytree = space['colsample_bytree'],
    learning_rate = space['learning_rate'],
    reg_alpha = space['reg_alpha'])

    scenarios_list = [ (1, 100, 50,100)]
    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    standard_xgboost_predictor(set_kernel = set_per_kernel),
    data_accumulator())

    x = scenario_generator_.accumulator.get_xs()[0]
    fx = scenario_generator_.accumulator.get_fxs()[0]
    z = scenario_generator_.accumulator.get_zs()[0]
    fz = scenario_generator_.accumulator.get_fzs()[0]
    eval_set  = [(x, fx), ( z, fz)]

    (_, registered_pred) = fit_and_predict(x,fx,z, model, 'registered_log')
    (_, casual_pred) = fit_and_predict(x,fx,z, model, 'casual_log')

    y_pred = (np.exp2(registered_pred) - 1) + (np.exp2(casual_pred) -1)
    
    score = get_relative_mean_squared_error(fz, y_pred)
    print(score)
    return{'loss':score, 'status': STATUS_OK }

def post_pred(y_pred):
    y_pred[y_pred < 0] = 0
    return y_pred

def rmsle(y_true, y_pred, y_pred_only_positive=True):
    if y_pred_only_positive: y_pred = post_pred(y_pred)
        
    diff = np.log(y_pred+1) - np.log(y_true+1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)    

def fit_and_predict(x,fx,z, model, target_variable):
    model.fit(x, fx)
    y_pred = model.predict(z)
    return (z, y_pred)

def test():
    from hyperopt import hp
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

    space ={
        'max_depth': hp.quniform("max_depth", 2, 50, 1),
        'n_estimators': hp.quniform("n_estimators", 1, 1000, 1),
        'subsample': hp.uniform ('subsample', 0.8, 1), 
        'colsample_bytree': hp.uniform ('colsample_bytree', 0.1, 1), 
        'learning_rate': hp.uniform ('learning_rate', 0.01, 0.1), 
        'reg_alpha': hp.uniform ('reg_alpha', 0.1, 1)
    }
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=15,
                trials=trials)

    print(best)
    return best
    


if __name__ == "__main__":
    set_per_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)
    def my_fun(x):
        import numpy as np
        from math import pi
        D = len(x)
        res = 1.
        for d in range(0,D):
            res *= np.cos(4 * x[d] * pi) 
        for d in range(0,D):
            res += x[d]
        return res

    opt_param = test()

    scenarios_list = [ (1, 100*i, 50,100*i ) for i in np.arange(1,5,1)]
    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,
    data_random_generator(fun = my_fun),
    standard_xgboost_predictor(set_kernel = set_per_kernel),
    data_accumulator(), **opt_param)
    results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
    print(results)
    list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 2)
