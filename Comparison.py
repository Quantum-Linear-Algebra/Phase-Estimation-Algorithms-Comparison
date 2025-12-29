

from numpy import pi

import Parameters as param

import sys 
paths = ['./0-Data', './1-Algorithms', './2-Graphing']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)
import Data_Manager as data
import Algorithm_Manager as algo
import Graph_Manager as graph_gen

# NOTE: Specifying unused parameters will not affect computation with the used parameters
parameters = {}

# System Parameters
parameters['comp_type']    = 'S' # OPTIONS: Classical, Simulation, Hardware, Job
parameters['sites']        = 2
parameters['scaling']      = 3/4*pi
parameters['shifting']     = 0
# parameters['overlap']      = 1   # the initial state overlap
# parameters['distribution'] = [.5]+[.5/(2^2-1)]*(2^2-1)
parameters['mod_ht']       = False
parameters['debugging']    = True 

# SPECIFIC SYSTEM TYPE
parameters['system'] = 'TFI' # OPTIONS: TFIM, SPIN, HUBBARD, H_2

# Transverse Field Ising Model Parameters
parameters['g']                = 4 # magnetic field strength (TFIM)
parameters['method_for_model'] = 'Q' # OPTIONS: F3C, Qiskit, Trotter
# parameters['trotter']          = 2 # only used with method_for_model = F3C, Trotter

# Spin Model Parameters
parameters['J'] = 0 # coupling strength (SPIN)

# Hubbard Parameters
parameters['t'] = 1 # left-right hopping (HUBB)
parameters['U'] = 10 # up-down hopping (HUBB)
parameters['x'] = 2 # x size of latice (HUBB)
parameters['y'] = 1 # y size of latice (HUBB)

# H_2 Parameters
parameters['distance'] = .5

# General Algorithm Paramters
parameters['max_T']         = 2
parameters['shots']         = 10**5
parameters['max_queries']   = 100 * 10**5
# parameters['num_time_sims'] = 1
# parameters['num_obs_sims']  = 1
parameters['reruns']        = 4

# NOTE: any parameters not filled out correctly will be set to default values (check displayed parameters)
parameters['algorithms']    = {}

parameters['algorithms']['ODMD'] = {}
parameters['algorithms']['ODMD']['svd_threshold']   = 10**-1
parameters['algorithms']['ODMD']['full_observable'] = False

# parameters['algorithms']['FDODMD'] = {}
# parameters['algorithms']['FDODMD']['svd_threshold']   = 10**-1
# parameters['algorithms']['FDODMD']['full_observable'] = True
# parameters['algorithms']['FDODMD']['gamma_range']     = (1,4) # (min, max)
# parameters['algorithms']['FDODMD']['filter_count']    = 4

# parameters['algorithms']['VQPE'] = {}
# parameters['algorithms']['VQPE']['svd_threshold']     = 10**-1
# parameters['algorithms']['VQPE']['T']                 = 40

parameters['algorithms']['UVQPE'] = {}
parameters['algorithms']['UVQPE']['svd_threshold']    = 10**-1
parameters['algorithms']['ODMD']['full_observable'] = True

parameters['algorithms']['QCELS'] = {}

# parameters['algorithms']['ML_QCELS'] = {}
# parameters['algorithms']['ML_QCELS']['time_steps']    = 5

# parameters['algorithms']['QMEGS'] = {}
# parameters['algorithms']['QMEGS']['sigma']            = 1
# parameters['algorithms']['QMEGS']['q']                = 0.05
# parameters['algorithms']['QMEGS']['alpha']            = 5
# parameters['algorithms']['QMEGS']['K']                = 2
# parameters['algorithms']['QMEGS']['full_observable']  = True
# parameters['algorithms']['QMEGS']['T']                = 100000
# parameters['algorithms']['QMEGS']['queries']          = 1000 * 10**2

if __name__ == "__main__":
    returns = param.check(parameters)
    data.run(parameters, returns)
    algo.run(parameters, skipping=parameters['max_queries']/parameters['shots']/20)
    graph_gen.run(parameters, show_std=True)