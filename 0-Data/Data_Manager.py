from qiskit_ibm_runtime import SamplerV2 as Sampler, EstimatorV2 as Estimator, OptionsV2 as Options
import pickle
import sys
sys.path.append('.')
from Service import empty, create_service
from Parameters import make_filename
sys.path.append('./1-Algorithms/Algorithms')

import subprocess, os, numpy as np
from scipy.linalg import expm, eigh, svd, norm
from scipy.fft import fft, ifft

from qiskit import transpile
from qiskit import qpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import UnitaryGate, StatePreparation

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# from qiskit_nature.second_q.drivers import PySCFDriver
# from qiskit_nature.second_q.mappers import ParityMapper

from QMEGS import generate_ts_distribution
from Parameters import check_contains_linear

# Prevent annoying migration warnings
import warnings 
warnings.simplefilter("ignore")

def closest_exp_unitary(A, t):
    """ 
    Description: Calculate the unitary matrix U that is closest with respect to the
    operator norm distance to the general matrix A. Used when qiskit fails to transpile
    unitary gate due to float point rounding.

    Args: Unitary matrix which qiskit fails to diagonalize: A

    Return: Unitary as an np matrix
    """
    A = -1j*A*t
    max_mag = 0
    for j in range(len(A)):
        for k in range(len(A[j])):
            mag = abs(A[j][k])
            if max_mag < mag:
                max_mag = mag
    if max_mag > 10**17:
        scale = 10**17/max_mag
        for j in range(len(A)):
            for k in range(len(A[j])):
                A[j][k] *= scale
    A = expm(A)
    V, __, Wh = svd(A)
    U = np.matrix(V.dot(Wh))
    return U

def create_hamiltonian(parameters, scale=True, show_steps=False):
    '''
    Create a system hamiltonian for the following systems:
     - Tranverse Field Ising Model (TFI)
     - Heisenberg Spin Model (SPI)
     - Hubbard Model (HUB)
     - Dihydrogen (H_2)

    Parameters:
     - parameters: a dictionary of parameters for contructing
       the Hamiltonian containing the following information
        - system: the system written as either: TFI, SPI, HUB, H_2
        - sites: the number of sites, default is 2
        - scaling: scales the eigenvalues to be at most this number
        - shifting: shift the eigenvalues by this value
        system specific parameters:
        TFI
        - g: magnetic field strength
        SPI
        - J: coupling strength
        HUB
        - t: left-right hopping
        - U: up-down hopping
        - x: x size of latice
        - y: y size of latice
        H_2
        - distance: the distance between two Hydrogen
     - show_steps: if true then debugging print statements
                   are shown
    
    Effects:
       This method also creates parameter['r_scaling'] which
       is used for recovering the original energy after ODMD.
     
    Returns:
     - H: the created hamiltonian
     - real_H_0: the minimum energy of the unscaled system
    '''

    system = parameters['system']
    scale_factor = parameters['scaling']
    shifting = parameters['shifting']
    if 'sites' in parameters.keys(): qubits = parameters['sites']
    else: qubits = 2
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    if system == "TFI":
        g = parameters['g']
        # construct the Hamiltonian
        # with Pauli Operators in Qiskit ^ represents a tensor product
        if show_steps: print("H = ", end='')
        for i in range(qubits-1):
            temp = Pauli('')
            for j in range(qubits):
                if (j == i or j == i+1):
                    temp ^= Pauli('Z')
                else:
                    temp ^= Pauli('I')
            H += -temp.to_matrix()
            if show_steps: print("-"+str(temp)+" ", end='')
        # peroidic bound
        # temp = Pauli('')
        # for j in range(qubits):
        #     if (j == 0 or j == qubits-1):
        #         temp ^= Pauli('Z')
        #     else:
        #         temp ^= Pauli('I')
        # H += -temp.to_matrix()
        if show_steps: print("-"+str(temp)+" ", end='')
        for i in range(qubits):
            temp = Pauli('')
            for j in range(qubits):
                if (j == i):
                    temp ^= Pauli('X')
                else:
                    temp ^= Pauli('I')
            H += -g*temp.to_matrix()
            if show_steps: print("-"+str(g)+"*"+str(temp)+" ", end='')
        if show_steps: print("\n")
    elif system == "SPI":
        qubits = parameters['sites']
        J = parameters['J']
        def S(index, coupling):
            temp = Pauli('')
            for j in range(qubits):
                if j == index:
                    temp ^= Pauli(coupling)
                else:
                    temp ^= Pauli('I')
            return 1/2*temp.to_matrix()
        if show_steps: print("H = ", end='\n')
        for qubit in range(qubits-1):
            H += S(qubit, 'X')@S(qubit+1, 'X')
            H += S(qubit, 'Y')@S(qubit+1, 'Y')
            H += S(qubit, 'Z')@S(qubit+1, 'Z')
        H += S(qubits-1, 'X')@S(0, 'X')
        H += S(qubits-1, 'Y')@S(0, 'Y')
        H += S(qubits-1, 'Z')@S(0, 'Z')
        H *= J
        if show_steps: print(H)
    elif system == "HUB":
        qubits = parameters['sites']
        x = parameters['x']
        y = parameters['y']
        U = parameters['U']
        t = parameters['t']       
        # coupling portion
        Sd = np.array([[0,0],[1,0]])
        S = np.array([[0,1],[0,0]])
        I = np.eye(2)
        left_right_hopping_term = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
        for op in [Sd]:
            for site in range(qubits):
                curr_x = site%x
                curr_y = site//x%y
                # couple sites in square latice
                neighbors = []
                if curr_x != 0:   neighbors.append((site-1)%qubits)
                if curr_x != x-1: neighbors.append((site+1)%qubits)
                if curr_y != 0:   neighbors.append((site+x)%qubits)
                if curr_y != y-1: neighbors.append((site-x)%qubits)
                for neighbor in neighbors:
                    temp = [1]
                    for site_ in range(qubits):
                        if site_ == site: temp = np.kron(temp, op)
                        elif site_ == neighbor: temp = np.kron(temp, op.T)
                        else: temp = np.kron(temp, I)
                    left_right_hopping_term+=temp
        left_right_hopping_term *=-t
        # number operator portion
        op1 = np.kron(Sd, Sd)
        op2 = np.kron(S, S)
        num = op1@op2
        up_down_hopping_term = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
        for place in range(qubits-1):
            temp = [1]
            for index in range(qubits-1):
                if index == place: temp = np.kron(temp, num) 
                else: temp = np.kron(temp, I) 
            up_down_hopping_term+=temp
        up_down_hopping_term*=U

        H = up_down_hopping_term+left_right_hopping_term
    elif system == 'H_2':
        distance = parameters['distance']
        parameters['sites'] = 1
        driver = PySCFDriver(
            atom=f'H .0 .0 .0; H .0 .0 {distance}',
            basis='sto3g'
        )
        molecule = driver.run()
        mapper = ParityMapper()
        fer_op = molecule.hamiltonian.second_q_op()
        tapered_mapper = molecule.get_tapered_mapper(mapper)
        H = tapered_mapper.map(fer_op)
        H = H.to_matrix()

    val, vec = eigh(H)
    real_E_0 = val[0]

    if scale:
        if show_steps:
            print("Original eigenvalues:", val)
            print("Original eigenvectors:\n", vec)
            print("Original Matrix:")
            for i in H:
                for j in i:
                    print(j, end = '\t\t')
                print()
        # # calculate the max magnitute eigenvalue (not correct yet)
        # max_iter = 1000
        # tol = 10**-10
        # n = H.shape[0]
        # v = np.random.rand(n) + 1j * np.random.rand(n)  # Initial random complex vector
        # lambda_old = 0
        # for _ in range(max_iter):
        #     v_next = H @ v
        #     print(v)
        #     lambda_new = np.vdot(v_next, v) / np.vdot(v, v)  # Rayleigh quotient
        #     if abs(lambda_new - lambda_old) < tol: break
        #     v = v_next
        #     lambda_old = lambda_new
        # scale eigenvalues of the Hamiltonian
        n = 2**qubits
        largest_eigenvalue = np.max(abs(val)) # use lambda_new when the above code segment
        if show_steps: print("Largest Eigenvalue =", largest_eigenvalue)
        parameters["r_scaling"] = largest_eigenvalue/scale_factor
        H *= scale_factor/largest_eigenvalue
        H += shifting*np.eye(n)
        if show_steps:
            val, vec = eigh(H)
            print("Scaled eigenvalues:", val)
            print("Scaled eigenvectors:\n", vec)
            min_eigenvalue = np.min(val)
            print("Lowest energy eigenvalue", min_eigenvalue); print()
    return H, real_E_0

# def run_hadamard_tests(controlled_U, statevector, W = 'Re', shots=100):
#     '''
#     Run a transpiled hadamard tests quantum circuit.

#     Parameters:
#      - controlled_U: the control operation to check phase of
#      - statevector: a vector to initalize the statevector of
#                     eigenqubits
#      - W: what type of hadamard tests to use (Re or Im)
#      - shots: the number of shots to run the tests with 

#     Returns:
#      - re: the real part of expection value measured
#     '''
#     aer_sim = AerSimulator(noise_model=NoiseModel())
#     trans_qc = create_hadamard_tests(aer_sim, controlled_U, statevector, W = W)
#     counts = aer_sim.run(trans_qc, shots = shots).result().get_counts()
#     exp_val = calculate_exp_vals(counts, shots)
#     return exp_val

# def create_hadamard_tests(parameters, backend, U:UnitaryGate, statevector, W = 'Re', modified=True):
#     '''
#     Creates a transpiled hadamard tests for the specificed backend.

#     Parameters:
#      - backend: the backend to transpile the circuit on
#      - controlled_U: the control operation to check phase of
#      - statevector: a vector to initalize the statevector of
#                     eigenqubits
#      - W: what type of hadamard tests to use (Re or Im)
    
#     Returns:
#      - trans_qc: the transpiled circuit
#     '''
#     qubits = parameters['sites']
#     qr_ancilla = QuantumRegister(1)
#     qr_eigenstate = QuantumRegister(qubits)
#     cr = ClassicalRegister(1)
#     qc = QuantumCircuit(qr_ancilla, qr_eigenstate, cr)
#     qc.h(qr_ancilla)
#     if modified:
#         qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
#         if parameters['system'] == 'TFI':
#             if parameters['g'] < 1:
#                 # construct GHZ state
#                 qc_init.ch(qr_ancilla,qr_eigenstate[0])
#                 for qubit in range(1, qubits):
#                     qc_init.cx(qubit, qubit+1)
#             else:
#                 # construct even superposition
#                 for qubit in range(1, qubits+1):
#                     qc_init.ch(qr_ancilla, qubit)
#         else:
#             gate = StatePreparation(statevector)
#             qc_init = qc_init.compose(gate.control(annotated="yes"))
        
#         qc = qc.compose(qc_init)
#         qc = qc.compose(U, range(1, qubits+1))
#         qc.x(0)
#         qc = qc.compose(qc_init)
#         qc.x(0)

#         phase = np.log(complex(U.to_matrix()[0][0])).imag
#         qc.rz(phase, qr_ancilla)
#     else:
#         qc.initialize(statevector, qr_eigenstate)
#         controlled_U = U.control(annotated="yes")
#         qc.append(controlled_U, qargs = [qr_ancilla] + qr_eigenstate[:])
    
#     if W[0:2].upper() == 'IM' or W[0].upper() == 'S': qc.sdg(qr_ancilla)
#     qc.h(qr_ancilla)
#     qc.measure(qr_ancilla[0],cr[0])
#     trans_qc = transpile(qc, backend, optimization_level=3)
#     return trans_qc

def create_hadamard_tests(parameters, backend, U:UnitaryGate, statevector=[], W = 'Re', modified=True):
    '''
    Creates a transpiled hadamard tests for the specificed backend.

    Parameters:
     - backend: the backend to transpile the circuit on
     - controlled_U: the control operation to check phase of
     - statevector: a vector to initalize the statevector of
                    eigenqubits
     - W: what type of hadamard tests to use (Re or Im)
     - modified: uses the modified hadamard test if true
    
    Returns:
     - trans_qc: the transpiled circuit
    '''
    qubits = parameters['sites']
    qr_ancilla = QuantumRegister(1)
    qr_eigenstate = QuantumRegister(qubits)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr_ancilla, qr_eigenstate, cr)
    qc.h(qr_ancilla)
    if modified:
        # print(U.to_matrix(), '\n')
        U_mat = U.to_matrix()
        import scipy; phase = scipy.linalg.logm(U_mat)[0][0].imag
        phase = np.log(complex(U_mat[0][0])).imag
        # print(phase, '\n')
        qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
        if parameters['system'] == 'TFI':
            if parameters['g'] < 1:
                # construct GHZ state
                qc_init.ch(qr_ancilla,qr_eigenstate[0])
                for qubit in range(1, qubits):
                    qc_init.cx(qubit, qubit+1)
            else:
                # construct even superposition
                for qubit in range(1, qubits+1):
                    qc_init.ch(qr_ancilla, qubit)
        else:
            gate = StatePreparation(statevector)
            qc_init = qc_init.compose(gate.control(annotated="yes"))
        
        qc = qc.compose(qc_init)
        qc = qc.compose(U, range(1, qubits+1))
        qc.x(0)
        qc = qc.compose(qc_init)
        qc.x(0)
        qc.rz(phase, qr_ancilla)
    else:
        # print(U.to_matrix(), '\n')

        qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
        if parameters['system'] == 'TFI':
            if parameters['g'] < 1:
                # construct GHZ state
                qc_init.h(qr_eigenstate[0])
                for qubit in range(1, qubits):
                    qc_init.cx(qubit, qubit+1)
            else:
                # construct even superposition
                for qubit in range(1, qubits+1):
                    qc_init.h(qubit)
        else:
            gate = StatePreparation(statevector)
            qc_init = qc_init.compose(gate, qubits=qr_eigenstate)
        
        qc = qc.compose(qc_init)
        controlled_U = U.control(annotated="yes")
        qc.append(controlled_U, qargs = [qr_ancilla] + qr_eigenstate[:])
    
    if W[0:2].upper() == 'IM' or W[0].upper() == 'S': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla[0],cr[0])
    # print(qc)
    trans_qc = transpile(qc, backend, optimization_level=3)
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=parameters['sites']+1)
    isa_observable = observable.apply_layout(trans_qc.layout)
    return (trans_qc, isa_observable)


def calculate_exp_vals(counts, shots):
    '''
    Calculates the real or imaginary of the expectation
    value depending on if the counts provided are from
    the real or the imaginary Hadamard tests.

    Parameters:
     - counts: the count object returned from result
     - shots: the number of shots to run the tests with 

    Returns:
     - meas: the desired expection value
    '''
    p0 = 0
    if counts.get('0') is not None:
        p0 = counts['0']/shots
    meas = 2*p0-1
    return meas

def make_overlap(ground_state, p):
    '''
    Creates a statevector with p probability of overlap with the ground state.

    Parameters:
     - groundstate: the ground statevector
     - p: the desired probability of overlap with the groundstate
    
    Returns:
     - phi: a statevector with the conditions
    '''
    length = len(ground_state)
    # generate a random vector with no overlap with the ground state
    random_vec = np.random.randn(length) + 1j * np.random.randn(length)
    random_vec -= np.vdot(ground_state, random_vec) * ground_state
    random_vec /= np.linalg.norm(random_vec)
    # contruct a statevector with p prob of overlap with groundstate
    phi = np.sqrt(p) * ground_state + np.sqrt(1 - p) * random_vec
    return phi

def rz_on_target_for_ZZ(qc, q0, q1, alpha):
    # implements exp(-i * alpha * Z⊗Z) using CNOT-Rz-CNOT
    # This implements exp(-i * alpha Z⊗Z) by doing Rz(2*alpha) on target with CNOT sandwich.
    # ---CNOT---
    # qc.rz(-np.pi/2,q0)
    # qc.rz(-np.pi,q1)
    # qc.sx(q1)
    # qc.rz(-np.pi, q1)
    # qc.ecr(q0,q1)
    # qc.x(q0)

    # # ---Rz---
    # qc.rz(2*alpha, q1)

    # # ---CNOT---
    # qc.rz(-np.pi/2,q0)
    # qc.rz(-np.pi,q1) # add rz rotation (2*alpha)
    # qc.sx(q1)
    # qc.rz(-np.pi,q1)
    # qc.ecr(q0,q1)
    # qc.x(q0)

    qc.cx(q0,q1)
    qc.rz(2*alpha, q1)
    qc.cx(q0,q1)

def rx_on(qc, q, beta):
    # implements exp(-i * beta * X) via Rx(2*beta) (Qiskit Rx angle φ implements exp(-i φ/2 X))
    # ---Rx---
    # qc.rz(np.pi/2,q)
    # qc.sx(q)
    # qc.rz(2*beta+np.pi, q)
    # qc.sx(q)
    # qc.rz(5*np.pi/2,q)
    qc.rx(2*beta, q)

def trotter_step_second_order(qc, j, g, dt, n_qubits):
    # positive j and g values to account for -t in time evolution
    # exponential for x
    beta_half = g * (dt/2)

    # exponential for zz
    alpha = j * dt

    # half X rotations
    for q in range(n_qubits):
        rx_on(qc, q, beta_half)

    # full ZZ
    for q in range(n_qubits - 1):
        rz_on_target_for_ZZ(qc, q, q+1, alpha)

    # half X rotations again
    for q in range(n_qubits):
        rx_on(qc, q, beta_half)

def trotter_evolution(j, g, n_qubits, t=1.0, r=1):
    # r = number of Trotter steps
    qc = QuantumCircuit(n_qubits)
    dt = t / r
    for _ in range(r):
        trotter_step_second_order(qc, j, g, dt, n_qubits)
    return qc.to_gate(label='trotter')

def hadamard_tests_circuit_info(parameters, T, observables, ML_QCELS=False, pauli_string='', gauss=[]):
    '''
    Gets information for creating exp_vals circuits. Creates controlled unitaries,
    and initialization statevector.

    Parameters:
     - T: the final time
     - parameters: the parmeters for the
                   hamiltonian contruction
    
    Returns:
     - gates: the controlled unitary gates
     - statevector: the initialization state vector
    '''
    statevector = parameters['sv']
    VQPE = pauli_string!=''
    QMEGS = len(gauss)!=0
    unordered_time_series = QMEGS or ML_QCELS
    use_F3C = not VQPE and parameters['system'] == 'TFI' and parameters['method_for_model'] == "F"
    use_Trotter = not VQPE and parameters['system'] == 'TFI' and parameters['method_for_model'] == 'T'

    if ML_QCELS:
        time_steps = set()
        iteration = 0
        time_steps_per_itr = parameters['algorithms']['ML_QCELS']['time_steps']
        while len(time_steps)+time_steps_per_itr < observables:
            for i in range(time_steps_per_itr):
                time = 2**iteration*i
                if time in time_steps: continue
                time_steps.add(time)
            iteration+=1
        time_steps = np.sort(list(time_steps))
        time_steps = [i*T/time_steps[-1] for i in time_steps]
    elif QMEGS:
        time_steps = gauss
    else:
        time_steps = [i*T/observables for i in range(observables)]
    
    Unitaries = []
    if use_F3C:
        coupling = 1
        scaling = parameters['scaling']
        sites = parameters['sites']
        g = parameters['g']
        trotter = parameters['trotter']
        if unordered_time_series:
            for time_step in time_steps:
                Unitaries.append(generate_TFIM_gates(sites, 1, time_step, g, scaling, coupling, trotter, '../f3cpp', include_0 = False)[0])
        else:
            Unitaries = generate_TFIM_gates(sites, observables, T/observables, g, scaling, coupling, trotter, '../f3cpp')
    elif use_Trotter:
        trotter = parameters['trotter']
        coupling, g = scale_TFIM_parameters(parameters)
        print(coupling, g)
        for time_step in time_steps:
            Unitaries.append(trotter_evolution(-coupling, -g, parameters['sites'], time_step, trotter))
    else:
        ham,_ = create_hamiltonian(parameters)
        Unitaries = []
        for i in time_steps:
            mat = closest_exp_unitary(ham,i)
            if VQPE:
                pauli = Pauli(pauli_string).to_matrix()
                mat = pauli@mat
            Unitaries.append(UnitaryGate(mat))
    return Unitaries, statevector

def generate_exp_vals(parameters, gausses={}):
    '''
    Generate the exp_vals spectrum

    Parameters:
     - parameters: the parmeters for the
                   hamiltonian contruction

    Returns:
     - exp_vals: the data generated
    '''

    sv = parameters['sv']
    H = parameters['Hamiltonian']
    E, vecs = eigh(H)
    spectrum = []
    for i in range(len(vecs)):
        spectrum.append(np.abs(sv.conj().T@vecs[:,i])**2)
    
    all_exp_vals = {}
    for time_series in parameters['time_series']:
        (time_series_name, T, observables, shots, full_observable) = time_series
        all_exp_vals[time_series] = [] 
        if time_series_name == 'linear':
            exp_vals = []
            for i in range(observables):
                exp_vals.append(np.sum(np.array(spectrum)*np.exp(-1j*E*i*T/observables)))
        if time_series_name == 'sparse':
            exp_vals = {}
            iteration = 0
            time_steps_per_itr = parameters['algorithms']['ML_QCELS']['time_steps']
            times = set()
            while len(times)+time_steps_per_itr < observables:
                for i in range(time_steps_per_itr):
                    times.add(2**iteration*i)
                iteration+=1
            for time in times:
                exp_vals[time] = np.sum(np.array(spectrum)*np.exp(-1j*E*time*T/max(times)))
        if time_series_name == 'vqpets':
            exp_vals = []
            length = observables
            if parameters['const_obs']: length = int(observables/((len(parameters['algorithms']['VQPE']['pauli_strings'])+1)))
            for i in range(length):
                exp_vals.append(np.sum(np.array(spectrum)*E*np.exp(-1j*E*i*T/observables)))
        if time_series_name == 'gausts':
            exp_vals = {}
            times = gausses[(T, observables)]
            for t in times:
                exp_vals[t] = np.sum(np.array(spectrum)*np.exp(-1j*E*t))
        all_exp_vals[time_series].append(exp_vals)
    return all_exp_vals

def transpile_hadamard_tests(parameters, T, observables, backend, W='Re', ML_QCELS=False, pauli_string='', gauss = []):
    '''
    Transpile the related hadamard tests to generate exp_vals

    Parameters:
     - T: the final evolution time
     - backend: the backend to transpile on
     - parameters: the parmeters for the hamiltonian
                   contruction

    Returns:
     - trans_qcs: the transpiled circuits
    '''

    tqcs = []
    Unitaries, statevector = hadamard_tests_circuit_info(parameters, T, observables, ML_QCELS=ML_QCELS, pauli_string=pauli_string, gauss=gauss)
    for U in Unitaries:
        tqcs.append(create_hadamard_tests(parameters, backend, U, statevector, W=W, modified=parameters['mod_ht']))
    return tqcs

def scale_TFIM_parameters(parameters):
    scaling = parameters['scaling']
    qubits = parameters['sites']
    g = parameters['g']
    coupling = 1
    # calculate new scaled parameters
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    for i in range(qubits-1):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i or j == i+1):
                temp ^= Pauli('Z')
            else:
                temp ^= Pauli('I')
        H += -temp.to_matrix()
    for i in range(qubits):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i):
                temp ^= Pauli('X')
            else:
                temp ^= Pauli('I')
        H += -g*temp.to_matrix()

    largest_eig = np.max(abs(np.linalg.eigvals(H)))
    coupling *= scaling/largest_eig
    g *= scaling/largest_eig
    return coupling, g

def generate_TFIM_gates(qubits, steps, dt, g, scaling, coupling, trotter, location, include_0 = True):
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    
    # calculate new scaled parameters
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    for i in range(qubits-1):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i or j == i+1):
                temp ^= Pauli('Z')
            else:
                temp ^= Pauli('I')
        H += -temp.to_matrix()
    for i in range(qubits):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i):
                temp ^= Pauli('X')
            else:
                temp ^= Pauli('I')
        H += -g*temp.to_matrix()

    largest_eig = np.max(abs(np.linalg.eigvals(H)))
    coupling *= scaling/largest_eig
    g *= scaling/largest_eig

    gates = []
    if not os.path.exists("TFIM_Operators"):
        os.mkdir("TFIM_Operators")
    
    # add timestep where dt = 0
    if include_0:
        with open("TFIM_Operators/Operator_Generator.ini", 'w+') as f:
            f.write("[Qubits]\nnumber = "+str(qubits)+"\n\n")
            f.write("[Trotter]\nsteps = 1\ndt = 0\n\n") # maybe need new number for steps
            f.write("[Jy]\nvalue = 0\n\n")
            f.write("[Jz]\nvalue = "+str(coupling)+"\n\n")
            f.write("[hx]\nramp = constant\nvalue = "+str(g)+"\n\n")
            f.write("[Output]\nname = TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=\nimin = 1\nimax = 2\nstep = 1\n")
        exe = location+"/release/examples/f3c_time_evolution_TFYZ"
        with open("TFIM_Operators/garbage.txt", "w") as file:
            subprocess.run([exe, "TFIM_Operators/Operator_Generator.ini"], stdout=file)
        os.remove("TFIM_Operators/garbage.txt")
        os.remove("TFIM_Operators/Operator_Generator.ini")
        qc = QuantumCircuit.from_qasm_file("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=1.qasm")
        gate = qc.to_gate(label = "TFIM 0")
        gates.append(gate)
        os.remove("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=1.qasm")
        steps -= 1

    steps *= trotter
    dt    /= trotter
    with open("TFIM_Operators/Operator_Generator.ini", 'w+') as f:
        f.write("[Qubits]\nnumber = "+str(qubits)+"\n\n")
        f.write("[Trotter]\nsteps = "+str(steps)+"\ndt = "+str(dt)+"\n\n") # maybe need new number for steps
        f.write("[Jy]\nvalue = 0\n\n")
        f.write("[Jz]\nvalue = "+str(coupling)+"\n\n")
        f.write("[hx]\nramp = constant\nvalue = "+str(g)+"\n\n")
        f.write("[Output]\nname = TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=\nimin = 1\nimax = "+str(steps+1)+"\nstep = 1\n")
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    with open("TFIM_Operators/garbage.txt", "w") as file:
        subprocess.run([exe, "TFIM_Operators/Operator_Generator.ini"], stdout=file)
    os.remove("TFIM_Operators/garbage.txt")
    os.remove("TFIM_Operators/Operator_Generator.ini")
    for step in range(1, steps+1):
        if step % trotter == 0:
            qc = QuantumCircuit.from_qasm_file("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i="+str(step)+".qasm")
            gate = qc.to_gate(label = "TFIM "+str(step))
            gates.append(gate)
        os.remove("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i="+str(step)+".qasm")
    os.rmdir("TFIM_Operators")
    return gates

import pickle, zipfile, io
from qiskit import qpy

def save_circuit(filename, trans_qcs):
    qcs = []
    obs = []
    for trans_qc in trans_qcs:
        (qc, ob) = trans_qc
        qcs.append(qc)
        obs.append(ob)
    with zipfile.ZipFile(filename+".zip", "w") as zf:
        buf = io.BytesIO() # make a virtual file in memory
        qpy.dump(qcs, buf)
        zf.writestr("circuits.qpy", buf.getvalue())
        zf.writestr("observables.pkl", pickle.dumps(obs))

def read_circuit(filename):
    with zipfile.ZipFile(filename+".zip", "r") as zf:
        buf = io.BytesIO(zf.read("circuits.qpy"))
        qcs = qpy.load(buf)
        obs = pickle.loads(zf.read("observables.pkl"))
    trans_qcs = []
    for i in range(len(qcs)):
        trans_qcs.append((qcs[i], obs[i]))
    return trans_qcs

def setup_circuit_data(parameters, time_series, optional_parameters):
    (time_series_name, T, observables, shots, full_observable) = time_series

    # randomly generate data for QMEGS if its needed
    gauss_distributed_ts = {}
    if time_series_name == 'gausts':
        gauss_distributed_ts[(T, observables)] = generate_ts_distribution(T, observables, parameters['algorithms']['QMEGS']['sigma'])
    optional_parameters['gauss_distributed_ts'] = gauss_distributed_ts

    if parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H':
        print('Circuits for a', time_series_name, ' time series with T =', T, ', observables =', observables, ', and shots =', shots, end='')
        if full_observable: print(' for both the Real and Imaginary Hadamard tests')
        else: print(' for only the Real Hadamard tests')

        try: os.mkdir('0-Data/Transpiled_Circuits')
        except: pass
        if time_series_name == 'vqpets':
            pauli_strings = parameters['algorithms']['VQPE']['pauli_strings']
            optional_parameters['pauli_strings'] = pauli_strings
            name = make_filename(parameters, T=T, obs=observables, fo=full_observable)
            for pauli_string in pauli_strings.paulis:
                pauli_string = str(pauli_string)

                filename = '0-Data/Transpiled_Circuits/'+pauli_string+'_'+name+'_Re.qpy'
                if empty(filename):
                    print('  Creating file for '+pauli_string+' Real Hadamard tests with observables =', observables)
                    trans_qcs = transpile_hadamard_tests(parameters, T, observables, optional_parameters['backend'], W='Re', pauli_string=pauli_string)
                    save_circuit(filename, trans_qcs)
                else:
                    print('  File found for '+pauli_string+' Imaginary Hadamard tests with observables =', observables)

                filename = '0-Data/Transpiled_Circuits/'+pauli_string+'_'+name+'_Im.qpy'
                if empty(filename):
                    print('  Creating file for '+pauli_string+' Imaginary Hadamard tests with observables =', observables)
                    trans_qcs = transpile_hadamard_tests(parameters, T, observables, optional_parameters['backend'], W='Im', pauli_string=pauli_string)
                    save_circuit(filename, trans_qcs)
                else:
                    print('  File found for '+pauli_string+' Imaginary Hadamard tests with  observables =', observables)
        else:
            name = make_filename(parameters, key=time_series_name, T=T, obs=observables, fo=full_observable)
            
            if time_series_name == 'sparse': ML_QCELS=True
            else: ML_QCELS = False
            if time_series_name == 'gausts':
                gauss_distributed_ts = optional_parameters['gauss_distributed_ts']
                gauss = gauss_distributed_ts[(T, observables)]
            else: gauss = []

            filename = '0-Data/Transpiled_Circuits/'+name+'_Re.qpy'
            if empty(filename):
                print('  Creating file for '+time_series_name+' Real Hadamard tests with observables =', observables)
                trans_qcs = transpile_hadamard_tests(parameters, T, observables, optional_parameters['backend'], W='Re', ML_QCELS=ML_QCELS, gauss=gauss)
                save_circuit(filename, trans_qcs)
            else:
                print('  File found for '+time_series_name+' Real Hadamard tests with observables =', observables)
            if full_observable:
                filename = '0-Data/Transpiled_Circuits/'+name+'_Im.qpy'
                if empty(filename):
                    print('  Creating file for '+time_series_name+' Imaginary Hadamard tests with observables =', observables)
                    trans_qcs = transpile_hadamard_tests(parameters, T, observables, optional_parameters['backend'], W='Im', ML_QCELS=ML_QCELS, gauss=gauss)
                    save_circuit(filename, trans_qcs)
                else:
                    print('  File found for Linear Imaginary Hadamard tests with observables =', observables)
        print()


def get_exp_vals_data(parameters, optional_parameters):
    results = []
    if parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H':
        trans_qcs = []
        for time_series in parameters['time_series']:
            (time_series_name, T, observables, shots, full_observable) = time_series
            print('Circuits for a', time_series_name, ' time series with T =', T, ', observables =', observables, ', and shots =', shots, end='')
            if full_observable: print(' for both the Real and Imaginary Hadamard tests')
            else: print(' for only the Real Hadamard tests')
            for run in range(parameters['reruns']):
                if time_series_name == 'vqpets':
                    for pauli_string in optional_parameters['pauli_strings'].paulis:
                        pauli_string = str(pauli_string)
                        name = make_filename(parameters, T=T, obs=observables, fo=full_observable)
                        filename = '0-Data/Transpiled_Circuits/'+pauli_string+'_'+name+'_Re.qpy'
                        print('  Loading data from file for '+pauli_string+' Real Hadamard tests for run', str(run+1)+'.')
                        trans_qcs.append(read_circuit(filename))
                        filename = '0-Data/Transpiled_Circuits/'+pauli_string+'_'+name+'_Im.qpy'
                        print('  Loading data from file for '+pauli_string+' Imaginary Hadamard tests for run', str(run+1)+'.')
                        trans_qcs.append(read_circuit(filename))
                else:
                    print('  Loading data from file for '+time_series_name+' Real Hadamard tests for run', str(run+1)+'.')
                    name = make_filename(parameters, key=time_series_name, T=T, obs=observables, fo=full_observable)
                    filename = '0-Data/Transpiled_Circuits/'+name+'_Re.qpy'
                    trans_qcs.append(read_circuit(filename))
                    if full_observable:
                        print('  Loading data from file for '+time_series_name+' Imaginary Hadamard tests for run', str(run+1)+'.')
                        filename = '0-Data/Transpiled_Circuits/'+name+'_Im.qpy'
                        trans_qcs.append(read_circuit(filename))
            if time_series_name == 'gausts':
                filename = '0-Data/Transpiled_Circuits/'+name+'_Re.qpy'
                os.remove(filename)
                if full_observable:
                    filename = '0-Data/Transpiled_Circuits/'+name+'_Im.qpy'
                    os.remove(filename)

        trans_qcs = sum(trans_qcs, []) # flatten list
        print()
        results = get_results(parameters, trans_qcs, optional_parameters['backend'])
        print('Data recieved.')
        print()
    elif parameters['comp_type'] == 'J':
        results = []
        service = create_service()
        for job_id in optional_parameters['job_ids']:
            print('Loading data from job:', job_id)
            job = service.job(job_id)
            for result in job.result():
                results.append(result)
            print('Loaded data from job:', job_id)
        create_hamiltonian(parameters)
        print()
    return results

def calculate_exp_vals_series(parameters, results, optional_parameters):
    gauss_distributed_ts = optional_parameters['gauss_distributed_ts']
    all_exp_vals = {}
    if parameters['comp_type'] == 'C':
        print('Generating Data')
        all_exp_vals = generate_exp_vals(parameters, gausses=gauss_distributed_ts)
    elif parameters['comp_type'] == 'S' or parameters['comp_type'] == 'H' or parameters['comp_type'] == 'J':
        print('Starting expecation value calculation for', len(results), 'circuit results.')
        
        time_series_list = list(parameters['time_series'])
        for ts_index in range(len(time_series_list)):
            time_series = time_series_list[ts_index]
            (time_series_name, T, observables, shots, full_observable) = time_series
            all_exp_vals[time_series] = []
            total_obs_per_run = 0
            for ts in time_series_list[0:ts_index]:
                num = ts[2]
                if ts[4]: num *= 2
                total_obs_per_run += num
            obs = observables
            if full_observable: obs *=2
            for r in range(parameters['reruns']):
                obs_used_this_itr = observables
                if full_observable: obs_used_this_itr *=2
                index = r*obs_used_this_itr + parameters['reruns']*total_obs_per_run
                if time_series_name == 'sparse':
                    list_exp_vals = calc_all_exp_vals(results[index:index+obs], shots)
                    time_steps = set()
                    iteration = 0
                    time_steps_per_itr = parameters['algorithms']['ML_QCELS']['time_steps']
                    while len(time_steps)+time_steps_per_itr < observables:
                        for j in range(time_steps_per_itr):
                            time = 2**iteration*j
                            if time in time_steps: continue
                            time_steps.add(time)
                        iteration+=1
                    time_steps = np.sort(list(time_steps))
                    exp_vals = {}
                    for j in range(len(time_steps)):
                        exp_vals[time_steps[j]] = list_exp_vals[j]
                elif time_series_name == 'vqpets':
                    pauli_strings = parameters['algorithms']['VQPE']['pauli_strings']
                    Hexp_vals = np.zeros(observables, dtype=complex)
                    for p in range(len(pauli_strings)):
                        start = index+p*observables
                        pauli_string = pauli_strings.paulis[p]
                        coeff = pauli_strings.coeffs[p]
                        exp_vals = calc_all_exp_vals(results[start:start+obs], shots)
                        Hexp_vals += [k*coeff for k in exp_vals]
                    exp_vals = Hexp_vals
                elif time_series_name == 'gausts':
                    exp_vals = {}
                    tmp = calc_all_exp_vals(results[index:index+obs], shots, fo=full_observable)
                    for j in range(len(tmp)):
                        exp_vals[gauss_distributed_ts[(T, observables)][j]] = tmp[j]
                else:
                    obs = observables
                    if full_observable: obs *=2
                    exp_vals = calc_all_exp_vals(results[index:index+obs], shots, fo = full_observable)
                assert(len(exp_vals)>0)
                all_exp_vals[time_series].append(exp_vals)
            assert(len(all_exp_vals[time_series])>0)
    print()
    return all_exp_vals

def get_results(parameters, trans_qcs, backend, retries=0):
    try:
        # options = Options()
        # options.shots = parameters['shots']
        estimator = Estimator(backend)

        if parameters['comp_type'] == 'H':
            job_correct_size = False
            jobs_tqcs = [trans_qcs]
            # Divide the circuits into multiple smaller jobs
            while(not job_correct_size):
                jobs = []
                job_correct_size = True
                for job_tqcs in jobs_tqcs:
                    print('Total shots in job:', len(job_tqcs)*parameters['shots'])
                    if len(job_tqcs)*parameters['shots']>=10000000: # shot limit
                        job_correct_size = False
                if job_correct_size:
                    try:
                        for tqcs in jobs_tqcs:
                            jobs.append(estimator.run(tqcs, shots = parameters['shots']))
                    except:
                        job_correct_size = False
                if not job_correct_size:
                    print('Job too large, splitting in half (max '+str(len(jobs_tqcs[0])//2)+' circuits per job)... ')
                    temp = []
                    for tqcs in jobs_tqcs:
                        half = int(len(tqcs)/2)
                        temp.append(tqcs[:half])
                        temp.append(tqcs[half:])
                    jobs_tqcs = temp
            print('Saving Parameters.')
            batch_id = jobs[0].job_id()
            job_ids = [job.job_id() for job in jobs]
            try: os.mkdir('0-Data/Jobs')
            except: pass
            with open('0-Data/Jobs/'+batch_id+'.pkl', 'wb') as file:
                pickle.dump([parameters, job_ids], file)
            print('Sending Job.')
        if parameters['comp_type'] == 'S':
            print('Running Circuits.')
            precision = 1/np.sqrt(parameters['shots'])
            jobs = [estimator.run(trans_qcs, precision = precision)]
        results = []
        for job in jobs:
            for result in job.result():
                if 'timing' in result.metadata:
                    print(f"Total Runtime Metadata: {result.metadata['timing']}")
                results.append(result) 
        return results
    except Exception as e:
        print(e)
        assert(retries<100)
        get_results(parameters, trans_qcs, backend, retries=retries+1)
        

def calc_all_exp_vals(results, shots, fo=True):
    num_timesteps = len(results)
    if fo: num_timesteps //= 2
    
    Res = []
    result = results[0:num_timesteps]
    for j in range(len(result)):
        raw_data = result[j].data
        Res.append(raw_data.evs)
    
    Ims = []
    if fo:
        start = num_timesteps
        result = results[start:(start+num_timesteps)]
        for j in range(len(result)):
            raw_data = result[j].data
            Ims.append(raw_data.evs)
    else:
        Ims = [0 for _ in range(len(Res))]
    
    exp_vals = []
    for i in range(len(Res)):
        ev = complex(Res[i], Ims[i])
        exp_vals.append(ev)
    return exp_vals

def save_exp_vals(parameters, all_exp_vals):
    try: os.mkdir('0-Data/Expectation_Values')
    except: pass
    for time_series in all_exp_vals:
        (time_series_name, T, observables, shots, full_observable) = time_series
        print(time_series, len(all_exp_vals[time_series][0]))
        assert(len(all_exp_vals[time_series][0])==observables)
        filename = '0-Data/Expectation_Values/'+make_filename(parameters, add_shots=True, shots=shots, key=time_series_name, T=T, obs=observables, fo=full_observable)+'.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(all_exp_vals[time_series], file)
        print('Saved expectation values into file.', '('+filename+')')

def run(parameters, returns):
    optional = {}
    if parameters['comp_type'] != 'C':
        backend = parameters['backend']
        optional['backend'] = backend
    if parameters['comp_type'] == 'J':
        job_ids = returns['job_ids']
        optional['job_ids'] = job_ids
    
    for time_series in parameters['time_series']:
        setup_circuit_data(parameters, time_series, optional)
    results = get_exp_vals_data(parameters, optional)
    all_exp_vals = calculate_exp_vals_series(parameters, results, optional)
    save_exp_vals(parameters, all_exp_vals)
    
# method for saving job ids when code was cancelled
def save_job_ids_params(parameters):
    job_ids = input('Enter Job ID(s):')
    job_ids = [job_ids[i*20:(i+1)*20] for i in range(len(job_ids)//20)]
    print(job_ids)
    with open('0-Data/Jobs/'+job_ids[0]+'.pkl', 'wb') as file:
        pickle.dump([parameters, job_ids], file)


if __name__ == '__main__':
    from Comparison import parameters
    from Parameters import check
    returns = check(parameters)
    run(parameters, returns)
    # save_job_ids_params(parameters)
