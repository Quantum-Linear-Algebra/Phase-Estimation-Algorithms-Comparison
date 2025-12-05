from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from scipy.linalg import eigh
from numpy import ceil, sqrt, zeros, log10, floor, abs, random, linspace
from scipy.linalg import norm
from Service import create_hardware_backend, empty
from sys import exit
import pickle
from qiskit.quantum_info import Operator, SparsePauliOp

def check(parameters):
    print('Setting up parameters.')
    
    # PREPROCESSING
    parameters['comp_type'] = parameters['comp_type'][0].upper()
    parameters['system']    = parameters['system'][0:3].upper()

    # parameter checking (if there's an error change parameters in question)
    assert(parameters['comp_type'] == 'C' or parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J')
    assert(parameters['system'] == 'TFI' or parameters['system'] == 'SPI' or parameters['system'] == 'HUB' or parameters['system'] == 'H_2')
    if 'overlap' in parameters: assert(0<=parameters['overlap']<=1)
    if 'distribution' in parameters: assert(0.9999999999999999<=sum(parameters['distribution'])<=1.0000000000000099) # rounding

    # verify system parameters are setup correctly
    returns = {}
    used_variables = []
    if parameters['comp_type'] == 'J':
        batch_id = input('Enter Job/Batch ID: ')
        print('Loading parameter data.')
        algos = parameters['algorithms']
        with open('0-Data/Jobs/'+str(batch_id)+'.pkl', 'rb') as file:
            [params, job_ids] = pickle.load(file)
        for key in params:
            used_variables.append(key)
            parameters[key] = params[key]
        parameters['algorithms'] = algos
        parameters['comp_type'] = 'J'
        returns['job_ids'] = job_ids
    else:
        used_variables = ['comp_type', 'sites', 'max_T', 'scaling', 'shifting', 'system',
                          'max_queries', 'r_scaling', 'const_obs', 'reruns', 'sv', 'shots',
                          'mod_ht']
        if 'debugging' in parameters and parameters['debugging']:
            import shutil
            if not empty('0-Data/Transpiled_Circuits'):
                shutil.rmtree("0-Data/Transpiled_Circuits/")
        if 'mod_ht' not in parameters: parameters['mod_ht'] = False
        parameters['max_T'] = float(parameters['max_T'])
        assert(parameters['max_T']>0)
        if 'shots' not in parameters: parameters['shots'] = 1 # parameters['comp_type'] == 'C' or 
        if parameters['comp_type'] == 'C' or 'reruns' not in parameters: parameters['reruns'] = 1
        if parameters['system'] == 'TFI':
            used_variables.append('g')
            if parameters['comp_type'] != 'C':
                used_variables.append('method_for_model')
                parameters['method_for_model'] = parameters['method_for_model'][0].upper()
                assert(parameters['method_for_model']=='F' or parameters['method_for_model']=='Q' or parameters['method_for_model']=='T')
                if parameters['method_for_model'] == 'F' or parameters['method_for_model']=='T': used_variables.append('trotter')
        elif parameters['system'] == 'HUB':
            used_variables.append('t')
            used_variables.append('U')
            x_in = 'x' in parameters.keys()
            y_in = 'y' in parameters.keys()
            if not x_in and not y_in:
                parameters['x'] = parameters['sites']
                parameters['y'] = 1
            elif not x_in: parameters['x'] = 1
            elif not y_in: parameters['y'] = 1
            x = parameters['x']
            y = parameters['y']
            assert(x>=0 and y>=0)
            assert(x*y == parameters['sites']) # change the latice shape
            used_variables.append('x')
            used_variables.append('y')
        elif parameters['system'] == 'SPI':
            used_variables.append('J')
            assert(parameters['J']!=0)
        elif parameters['system'] == 'H_2':
            used_variables.append('distance')
            parameters['sites']=1
        
        import sys
        sys.path.append('0-Data')
        from Data_Manager import create_hamiltonian, make_overlap
        H,real_E_0 =create_hamiltonian(parameters)
        used_variables.append('Hamiltonian')
        parameters['Hamiltonian'] = H
        used_variables.append('real_E_0')
        parameters['real_E_0'] = real_E_0
        energy,eig_vec = eigh(H)
        if 'overlap' in parameters:
            used_variables.append('overlap')
            if parameters['system'] == 'TFI':
                if parameters['g']< 1:
                    # GHZ state
                    sv = [0]*2**parameters['sites']
                    sv[0] = 1
                    sv[-1] = 1
                    sv = sv/norm(sv)
                else:
                    # construct even superposition
                    sv = [1]*2**parameters['sites']
                    sv = sv/norm(sv)
            else:
                sv = make_overlap(eig_vec[:,0], parameters['overlap'])
            print(sv)
            parameters['sv'] = sv
        elif 'distribution' in parameters:
            used_variables.append('distribution')
            parameters['sv'] = zeros(len(eig_vec[:,0]), dtype=complex)
            for i in range(len(parameters['distribution'])):
                # print(i, parameters['distribution'])
                parameters['sv'] += sqrt(parameters['distribution'][i])*eig_vec[:,i]
                # print(parameters['sv']@eig_vec[:,i])
            # assert(parameters['sv']@eig_vec[:,0]==parameters['distribution'][0]) 
        else: parameters['sv'] = eig_vec[:,0]
        used_variables.append('scaled_E_0')
        parameters['scaled_E_0'] = energy[0]
        
        if 'const_obs' not in parameters:
            parameters['const_obs'] = False

        # used_variables.append('final_times')
        # used_variables.append('final_observables')
        # if not parameters['const_obs']:
        #     num_sims = 10
        #     if 'num_time_sims' in parameters: num_sims = parameters['num_time_sims']
        #     parameters['final_times'] = linspace(0, parameters['max_T'], num_sims+1)[1:] # excluding 0
        #     num_sims = 10
        #     if 'num_obs_sims' in parameters: num_sims = parameters['num_obs_sims']
        #     parameters['final_observables'] = [int(i) for i in linspace(0, parameters['observables'], num_sims+1)[1:]] # excluding 0
        # else:
        #     parameters['final_times'] = [parameters['max_T']]
        #     parameters['final_observables'] = [parameters['observables']]
    
    used_variables.append('algorithms')
    for algo in parameters['algorithms']:
        assert(algo in ['VQPE','UVQPE','ODMD','FDODMD','QCELS','ML_QCELS','QMEGS'])

    # calculate lambda prior if needded
    if 'QCELS' in parameters['algorithms'] or 'ML_QCELS' in parameters['algorithms']:
        # Approximate what Hartree-Fock would estimate
        if 'lambda_prior' in parameters:
            lambda_prior = parameters['algorithms']['QCELS']['lambda_prior']
        else:
            E_0 = parameters['scaled_E_0']
            order = floor(log10(abs(E_0)))
            if 'lambda_digits' in parameters:
                digits = parameters['algorithms']['QCELS']['lambda_digits']
                if digits == -1: digits = int(random.randint(1,3))
            else: digits = 2
            lambda_prior = -(int(str(E_0*10**(-order+digits))[1:digits+1])+random.rand())*(10**(order-digits+1))

    if 'VQPE' in parameters['algorithms']:
        if 'svd_threshold' not in parameters['algorithms']['VQPE']: parameters['algorithms']['VQPE']['svd_threshold'] = 10**-6
        parameters['algorithms']['VQPE']['pauli_strings'] = SparsePauliOp.from_operator(Operator(H))
        # total_num_time_series = 2*(len(parameters['algorithms']['VQPE']['pauli_strings'])+1)
        # if parameters['const_obs'] and parameters['observables']%total_num_time_series!=0:
        #     parameters['observables'] = int(ceil(parameters['observables']/total_num_time_series)*total_num_time_series)
        #     for i in range(len(parameters['final_observables'])):
        #         parameters['final_observables'][i] = int(ceil(parameters['final_observables'][i]/total_num_time_series)*total_num_time_series)
    if 'QCELS' in parameters['algorithms']:
        parameters['algorithms']['QCELS']['lambda_prior'] = lambda_prior
    if 'ML_QCELS' in parameters['algorithms']:
        parameters['algorithms']['ML_QCELS']['lambda_prior'] = lambda_prior
        # make sure the time steps per iteration is defined
        if 'time_steps' not in parameters['algorithms']['ML_QCELS']: parameters['algorithms']['ML_QCELS']['time_steps'] = 5

        # iteration = 0
        # time_steps_per_itr = parameters['algorithms']['ML_QCELS']['time_steps']
        # times = set()
        # while len(times) < parameters['observables']/2:
        #     for i in range(time_steps_per_itr):
        #         times.add(2**iteration*i)
        #     iteration+=1
        # for obs in range(len(parameters['final_observables'])):
        #     iteration = 0
        #     time_steps_per_itr = parameters['algorithms']['ML_QCELS']['time_steps']
        #     times = set()
        #     while len(times) < parameters['final_observables'][obs]/2:
        #         for i in range(time_steps_per_itr):
        #             times.add(2**iteration*i)
        #         iteration+=1
        #     parameters['final_observables'][obs] = len(times)*2
        # if 'calc_Dt' in parameters and parameters['algorithms']['ML_QCELS']['calc_Dt']:
        #     delta = 1*sqrt(1-parameters['overlap'])
        #     parameters['max_T'] = parameters['observables']*delta/parameters['algorithms']['ML_QCELS']['time_steps']
    if 'ODMD' in parameters['algorithms']:
        if 'svd_threshold' not in parameters['algorithms']['ODMD']: parameters['algorithms']['ODMD']['svd_threshold'] = 10**-6
        if 'full_observable' not in parameters['algorithms']['ODMD']: parameters['algorithms']['ODMD']['full_observable'] = False
    if 'FDODMD' in parameters['algorithms']:
        if 'svd_threshold' not in parameters['algorithms']['FDODMD']: parameters['algorithms']['FDODMD']['svd_threshold'] = 10**-6
        if 'full_observable' not in parameters['algorithms']['FDODMD']: parameters['algorithms']['FDODMD']['full_observable'] = False
        if 'gamma_range' not in parameters['algorithms']['FDODMD']:
            parameters['algorithms']['FDODMD']['gamma_range'] = (1,3)
        else:
            assert(parameters['algorithms']['FDODMD']['gamma_range'][0]>=0 and parameters['algorithms']['FDODMD']['gamma_range'][1]>=0)
            assert(parameters['algorithms']['FDODMD']['gamma_range'][0]<=parameters['algorithms']['FDODMD']['gamma_range'][1])
        if 'filter_count' not in parameters['algorithms']['FDODMD']: parameters['algorithms']['FDODMD']['filter_count'] = 6
    if 'UVQPE' in parameters['algorithms']:
        if 'svd_threshold' not in parameters['algorithms']['UVQPE']: parameters['algorithms']['UVQPE']['svd_threshold'] = 10**-6
    if 'QMEGS' in parameters['algorithms']:
        if 'sigma' not in parameters['algorithms']['QMEGS']: parameters['algorithms']['QMEGS']['sigma'] = 0.5
        if 'q' not in parameters['algorithms']['QMEGS']: parameters['algorithms']['QMEGS']['q'] = 0.05
        if 'alpha' not in parameters['algorithms']['QMEGS']: parameters['algorithms']['QMEGS']['alpha'] = 5
        if 'K' not in parameters['algorithms']['QMEGS']: parameters['algorithms']['QMEGS']['K'] = 1
        if 'full_observable' not in parameters['algorithms']['QMEGS']: parameters['algorithms']['QMEGS']['full_observable'] = True

    used_variables.append('time_series')
    parameters['time_series'] = {}
    for algo in parameters['algorithms']:
        algo_params = parameters['algorithms'][algo]
        print(algo, algo_params)
        if 'T' in algo_params:
            T = algo_params['T']
        else:
            T = parameters['max_T']
        
        if 'shots' in algo_params:
            shots = algo_params['shots']
        else:
            shots = parameters['shots']
            
        if 'full_observable' in algo_params:
            full_observable = algo_params['full_observable']
        else:
            full_observable = True 
        
        if 'queries' in algo_params:
            queries = algo_params['queries']
        else:
            queries = parameters['max_queries']
        
        obs = queries//shots # just real 
        if full_observable:
            obs //= 2 # real and imaginary

        if algo == 'VQPE':
            time_dist = 'vqpets'
            obs //= len(algo_params['pauli_strings'])
            # check to see if theres a useable linear time series
            found = False
            for time_series in parameters['time_series']:
                (time_dist2, T2, obs2, shots2, fo2)  = time_series
                if time_dist2 == 'linear' and fo2 == full_observable and T/obs == T2/obs2 and shots==shots2:
                    found = True
                    break
            # make one if there isn't a useable one
            if not found:
                parameters['time_series'][('linear', T, obs, shots, full_observable)] = []
        elif check_contains_linear([algo]):
            time_dist = 'linear'
        elif algo == 'QMEGS':
            time_dist = 'gausts'
        elif algo == 'ML_QCELS':
            time_dist = 'sparse'
        time_series = (time_dist, T, obs, shots, full_observable)
        if time_series not in parameters['time_series']:
            parameters['time_series'][time_series] = []
        parameters['time_series'][time_series].append(algo)

    keys = []
    for i in parameters.keys():
        keys.append(i)
    for key in keys:
        if key not in used_variables:
            parameters.pop(key)

    # backend setup
    if parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J':
        parameters['backend'] = create_hardware_backend()
    elif parameters['comp_type'] == 'S':
        parameters['backend'] = AerSimulator(noise_model = NoiseModel())
    
    print('Parameters are setup:')
    for key in parameters:
        print('  '+key+':', parameters[key])
    print()
    return returns

# define a system for naming files
def make_filename(parameters, add_shots = False, key='', T = -1, obs=-1, shots=-1, fo=True):
    system = parameters['system']
    
    string = ''
    if key != '': string += key+'_'
    if parameters['comp_type'] != 'C':
        string += 'comp='+parameters['backend'].name
        string +='_mod_ht='+str(parameters['mod_ht'])[0]
    else: string += 'comp='+parameters['comp_type']
    string +='_sys='+system+'_n='+str(parameters['sites'])
    if system=='TFI':
        if parameters['comp_type'] != 'C':
            method_for_model = parameters['method_for_model']
            string+='_m='+method_for_model
            if method_for_model == 'F' or method_for_model == 'T':
                string+='_trotter='+str(parameters['trotter'])
        string+='_g='+str(parameters['g'])
    elif system=='SPI':
        string+='_J='+str(parameters['J'])
    elif system=='HUB':
        string+='_t='+str(parameters['max_T'])
        string+='_U='+str(parameters['U'])
        string+='_x='+str(parameters['x'])
        string+='_y='+str(parameters['y'])
    elif system=='H_2':
        string+='_dist='+str(parameters['distance'])
    string+='_scale='+str(parameters['scaling'])
    string+='_shift='+str(parameters['shifting'])
    if 'overlap' in parameters: string+='_overlap='+str(parameters['overlap'])
    if 'distribution' in parameters:
        string+='_distr=['
        for i in parameters['distribution'][:3][:-1]:
            string+=f'{i:0.2},'
        var = parameters['distribution'][3]
        string+=f'{var:0.2}]'
    if T == -1: string+='_T='+str(parameters['max_T'])
    else: string+='_T='+str(T)
    if obs == -1:
        if parameters['algorithms'] == ['VQPE'] and parameters['const_obs']:
            string += '_obs='+str(int(parameters['observables']/(len(parameters['algorithms']['VQPE']['pauli_strings'])+1)))
        else:
            string += '_obs='+str(parameters['max_queries']//parameters['shots'])
    else:
        string += '_obs='+str(obs)
    if key == 'gausts':
        string += '_sigma='+str(parameters['algorithms']['QMEGS']['sigma'])
    if add_shots:
        string += '_reruns='+str(parameters['reruns'])
        if parameters['comp_type'] != 'C':
            if shots!=-1:
                string += '_shots='+str(parameters['shots'])
            else:
                string += '_shots='+str(shots)
    if not fo:
        string += '_onlyRe'
    return string

def check_contains_linear(algos):
    linear = ['ODMD', 'FDODMD', 'VQPE', 'UVQPE', 'QCELS']
    for algo in algos:
        if algo in linear:
            return True
    return False
    

if __name__ == '__main__':
    from Comparison import parameters
    check(parameters)