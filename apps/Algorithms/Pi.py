from pathlib import Path
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = Path(dir_path).parent
sys.path.append(os.path.join(dir_path,"common"))
from codpy_tools import *
sys.path.append(os.path.join(dir_path,"Danske"))
from Bachelier import *


D, Nx,Ny,Nz = 2, 300,300,300
scenarios_list = [ (2, 2**(i-2), 2**(i-2), 2**(i-2))  for i in np.arange(8,12,1)]
print(scenarios_list)
data_ = data_generator_Bachelier_iid(seed1 = 42, seed2 = 35, seed3 = 37)

scenarios = scenario_generator()
scenarios.run_scenarios(scenarios_list,data_,codpyprRegressor(),data_accumulator())
results = scenarios.accumulator.get_output_datas()

# basketxs = data_.basket(x = scenarios.accumulator.get_xs())
# basketzs = data_.basket(x = scenarios.accumulator.get_zs())
# scenarios.accumulator.plot_learning_and_train_sets(basketxs,basketzs,labelx='Basket values')

print(results)

D, Nx,Ny,Nz = 5, 300,300,300
data_ = data_generator_Bachelier_sharp(seed1 = 42, seed2 = 35, seed3 = 37)
data_.set_data(D, Nx,Ny,Nz)
x,z,fx,fz = data_.x,data_.z,data_.fx,data_.fz

scenarios = scenario_generator()
scenarios.run_scenarios(scenarios_list,data_,codpyprRegressor(),data_accumulator())
results = scenarios.accumulator.get_output_datas()

print(results)