import numpy as np
pi = np.pi 
from Service import create_hardware_backend
import matplotlib.pyplot as plt

from scipy.linalg import expm, eigh, norm, eig

from qiskit import transpile
from qiskit.quantum_info import Pauli
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate, StatePreparation

from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler, EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService as QRS
from qiskit.quantum_info import Operator, Statevector 
from matplotlib.colors import LogNorm

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
        qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
        if len(statevector) == 0:
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

        ev = complex(U.to_matrix()[0][0])
        phase = np.log(ev)
        phase = phase.imag
        qc.rz(phase, qr_ancilla)
    else:
        qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
        if len(statevector) == 0:
            if parameters['g'] < 1:
                # construct GHZ state
                qc_init.ch(qr_ancilla[0], qr_eigenstate[0])
                for qubit in range(qubits):
                    qc_init.x(qr_eigenstate[qubit])
            else:
                # construct even superposition
                for qubit in range(1, qubits+1):
                    qc_init.h(qubit)
        else:
            gate = StatePreparation(statevector)
            qc_init = qc_init.compose(gate.control(annotated="yes"))
        qc = qc.compose(qc_init)
        controlled_U = U.control(annotated="yes")
        qc.append(controlled_U, qargs = [qr_ancilla] + qr_eigenstate[:])
    
    if W[0:2].upper() == 'IM' or W[0].upper() == 'S': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla[0],cr[0])
    # print(qc)
    # trans_qc = transpile(qc, backend, optimization_level=3)
    trans_qc = transpile(qc, optimization_level=3, basis_gates=['id','ecr','rz','sx','x'])
    return trans_qc

def create_trot_ht(parameters, backend, trot_u_circ, statevector=[], W = 'Re', modified=True):
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
        qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
        if len(statevector) == 0:
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
        qc = qc.compose(trot_u_circ, range(1, qubits+1))
        qc.x(0)
        qc = qc.compose(qc_init)
        qc.x(0)

        ev = complex(Operator(trot_u_circ).data[0][0])
        phase = np.log(ev)
        phase = phase.imag
        qc.rz(phase, qr_ancilla)
    else:
        qc_init = QuantumCircuit(qr_ancilla, qr_eigenstate)
        if len(statevector) == 0:
            if parameters['g'] < 1:
                # construct GHZ state
                qc_init.ch(qr_ancilla[0], qr_eigenstate[0])
                for qubit in range(qubits):
                    qc_init.x(qr_eigenstate[qubit])
            else:
                # construct even superposition
                for qubit in range(1, qubits+1):
                    qc_init.h(qubit)
        else:
            gate = StatePreparation(statevector)
            qc_init = qc_init.compose(gate.control(annotated="yes"))
        qc = qc.compose(qc_init)
        controlled_U = U.control(annotated="yes")
        qc.append(controlled_U, qargs = [qr_ancilla] + qr_eigenstate[:])
    
    if W[0:2].upper() == 'IM' or W[0].upper() == 'S': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    # qc.measure(qr_ancilla[0],cr[0])
    # print(qc)
    # trans_qc = transpile(qc, backend, optimization_level=3)
    trans_qc = transpile(qc, optimization_level=3, basis_gates=['id','ecr','rz','sx','x'])
    return trans_qc

def create_hamiltonian(parameters, scale=True, show_steps=False):
    '''
    Create a system hamiltonian for the Tranverse Field Ising Model

    Parameters:
     - parameters: a dictionary of parameters for contructing
       the Hamiltonian containing the following information
        - sites: the number of sites, default is 2
        - scaling: scales the eigenvalues to be in [-scaling, scaling]
        - shifting: shift the eigenvalues by this value
        - g: magnetic field strength
     - show_steps: if true then debugging print statements
                   are shown
    
    Effects:
       This method also creates parameter['r_scaling'] which
       is used for recovering the original energy.
     
    Returns:
     - H: the created hamiltonian
     - real_H_0: the minimum energy of the unscaled system
    '''
    scale_factor = parameters['scaling']
    shifting = parameters['shifting']
    if 'sites' in parameters.keys(): qubits = parameters['sites']
    else: qubits = 2
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
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
    # if show_steps: print("-"+str(temp)+" ", end='')
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
    return H, real_E_0, largest_eigenvalue

def create_hardware_backend():
    '''
    Creates a hardware backend using the inputted Qiskit user data.

    Returns:
     - backend: the specificed backend as a BackendV2 Qiskit Object
    '''
    hardware_name = input("Enter Hardware Backend Name:")
    token    = input("Enter API Token:")
    instance = input("Enter Instance:")
    try:
        print("Creating backend.")
        service = QRS(channel='ibm_cloud', instance=instance, token=token)
        backend = service.backend(hardware_name)
        print("Backend created.")
        return backend
    except Exception as e:
        print(e)
        print("One or more of the provided service parameters are incorrect. Try again.")
        create_hardware_backend()

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
    return qc

def calculate_exp_vals(counts, shots):
    '''
    Calculates the real or imaginary of the expectation
    value depending on if the counts provided are from
    the real or the imaginary Hadamard tests.

    Parameters:
     - counts: the count object returned from result
     - shots: the number of shots used to run the tests with 

    Returns:
     - meas: the desired expection value
    '''
    p0 = 0
    if counts.get('0') is not None:
        p0 = counts['0']/shots
    meas = 2*p0-1
    return meas

if __name__ == '__main__':
    use_hardware = False
    if use_hardware:
        backend = create_hardware_backend()
    else:
        backend = AerSimulator()
    sampler = Sampler(backend)

    t = 1
    # shots = 1000
    # Trotter circuit
    # trot_j = parameters['scaling']/E_L
    # trot_g = parameters['g']*parameters['scaling']/E_L

    sites_list = np.arange(2,8,1)
    # trot_list = [int((x//2*2)+1) for x in np.linspace(2,14,len(sites_list)).astype(int)]
    trot_list = np.arange(2,10,1).tolist()

    print('sites list', sites_list)
    print('trot list', trot_list)

    norm_diffs = []
    # depths_2q = []
    # qiskit_depths = []
    spectral_diffs = []
    spectral_sorted_diffs = []
    for i in range(len(sites_list)):
        norm_diffs.append([])
        spectral_diffs.append([])
        spectral_sorted_diffs.append([])

        # depths_2q.append([])
        print('Generating data for',sites_list[i],'sites')

        # create example circuit
        parameters = {}
        parameters['sites']    = sites_list[i]
        parameters['scaling']  = 3*pi/4
        parameters['shifting'] = 0
        parameters['g']        = 4 
        H, E_0, E_L = create_hamiltonian(parameters, show_steps=False)
        U = expm(-1j*H*t)
        U_eigval, _ = eig(U)
        U_phases = [-np.log(eigval).imag for eigval in U_eigval]
        U_phases_sorted = np.sort(U_phases)
        # print(U_phases)
        # qc_qiskit = create_hadamard_tests(parameters, backend, U, modified=True)
        # trans_qc_qiskit = transpile(qc_qiskit, optimization_level=3, basis_gates=['id','ecr','rz','sx','x']) # just rpi_rensselaer basis gates
        # qiskit_depths.append(qc_qiskit.count_ops().get('ecr'))
        trot_j = parameters['scaling']/E_L
        trot_g = parameters['g']*parameters['scaling']/E_L
        for j in range(len(trot_list)):
            print('  trotter steps:', trot_list[j])
            qc_trot_unitary = trotter_evolution(-trot_j, -trot_g, sites_list[i], t=t, r=trot_list[j]) # increase r to reduce Trotter error
            trans_qc_trot = transpile(qc_trot_unitary, optimization_level=3, basis_gates=['id','ecr','rz','sx','x']) # just rpi_rensselaer basis gates
            trot_mat = Operator(trans_qc_trot).data
            norm_diffs[i].append(np.linalg.norm(trot_mat-U, ord=2))

            trot_eigval, _ = eig(trot_mat)
            trot_phases = [-np.log(eigval).imag for eigval in trot_eigval]
            
            sum = 0
            for phase_index in range(len(U_phases)):
                sum += abs(trot_phases[phase_index]-U_phases[phase_index])
            spectral_diffs[i].append(sum)

            trot_phases_sorted = np.sort(trot_phases)
            sum = 0
            for phase_index in range(len(U_phases_sorted)):
                sum += abs(trot_phases_sorted[phase_index]-U_phases_sorted[phase_index])
            spectral_sorted_diffs[i].append(sum/len(U_phases_sorted))
            
            # qc_trot = create_trot_ht(parameters, backend, qc_trot_unitary, modified=True)
            # qiskit_mat = Operator(qc_qiskit).data
            # trans_qc_trot = transpile(qc_trot, optimization_level=3, basis_gates=['id','ecr','rz','sx','x']) # just rpi_rensselaer basis gates
            # print(' gate counts:', trans_qc_trot.count_ops())
            # depths_2q[i].append(trans_qc_trot.count_ops().get('ecr'))
        
    # print(norm_diffs)
    # print(spectral_diffs)
    # print(depths_2q)
    # for i in range(len(depths_2q)): depths_2q[i].append(qiskit_depths[i])

    fig, axes = plt.subplots(1, 2, figsize=(2*len(trot_list),len(sites_list)))

    colorbar_scale = LogNorm(vmin=10**0, vmax=10**-5)
    im1 = axes[0].imshow(norm_diffs, norm=colorbar_scale)#LogNorm(vmin=np.min(norm_diffs), vmax=np.max(norm_diffs)))
    for ax in axes:
        ax.set_xticks(range(len(trot_list)), labels=np.array(trot_list).astype(str),
                    rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(sites_list)), labels=np.array(sites_list).astype(str))

    # Loop over data dimensions and create text annotations.
    # for i in range(len(sites_list)):
    #     for j in range(len(trot_list)):
    #         axes[0].text(j, i, f'{np.log(norm_diffs[i][j]):1.2f}', ha="center", va="center", color="w")
    cbar = fig.colorbar(im1, ax=axes[0])
    cbar.set_label('2-norm difference')

    
    # # print(np.min(spectral_diffs), np.max(spectral_diffs))
    # spectrum_diff_ax = axes[1].imshow(spectral_diffs, norm=colorbar_scale)#LogNorm(vmin=np.min(spectral_diffs), vmax=np.max(spectral_diffs)))
    # cbar = fig.colorbar(spectrum_diff_ax, ax=axes[1])
    # cbar.set_label('Phase difference for entire spectrum')

    spectrum_sorted_diff_ax = axes[1].imshow(spectral_sorted_diffs, norm=colorbar_scale)#LogNorm(vmin=np.min(spectral_sorted_diffs), vmax=np.max(spectral_sorted_diffs)))
    cbar = fig.colorbar(spectrum_sorted_diff_ax, ax=axes[1])
    cbar.set_label('Phase difference for sorted spectrum')
    # axes[1].set_xticks(range(len(trot_list)), labels=np.array(trot_list).astype(str),
    #             rotation=45, ha="right", rotation_mode="anchor")
    # axes[1].set_yticks(range(len(sites_list)), labels=np.array(sites_list).astype(str))
    # depth graph
    # axes[1].set_title("Trotterized TFIM TE U err")
    # axes[1].set_xlabel('Trotter Steps')
    # axes[1].set_ylabel('TFIM Sites')
    # im2 = axes[1].imshow(depths_2q)
    # Show all ticks and label them with the respective list entries
    # trot_list.append('qiskit')
    # axes[1].set_xticks(range(len(trot_list)), labels=np.array(trot_list).astype(str),
    #             rotation=45, ha="right", rotation_mode="anchor")
    # axes[1].set_yticks(range(len(sites_list)), labels=np.array(sites_list).astype(str))
    # cbar = fig.colorbar(im2, ax=axes[1])
    # cbar.set_label('2-qubit gate counts')
    # axes[1].set_title("Modified HT TFIM 2-q gate counts")
    # axes[1].set_xlabel('Trotter Steps/Method')
    # axes[1].set_ylabel('TFIM Sites')
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(sites_list)):
    #     for j in range(len(trot_list)):
    #         text = axes[1].text(j, i, depths_2q[i][j],
    #                     ha="center", va="center", color="w")
    
    plt.savefig('testing_graphs/heatmap.pdf')
    plt.close()


    # qc_trot = trotter_evolution(-trot_j, -trot_g, parameters['sites'], t=t, r=r)   # increase r to reduce Trotter error
    # qc_trot.draw(output='mpl', filename='HT_site'+str(parameters['sites'])+'_trot.pdf')
    # qc_trot = transpile(qc_trot, optimization_level=3, basis_gates=['id','ecr','rz','sx','x']) # just rpi_rensselaer basis gates
    # # qc_trot = transpile(qc_trot, backend, optimization_level=3) # rpi_rensselaer basis gates + qpu mapping gates (swapping, etc.)
    # qc_trot.draw(output='mpl', idle_wires=False, filename='HT_opt3_site'+str(parameters['sites'])+'_trot.pdf')
    # print('Trotter unitary gate counts:', qc_trot.count_ops())
    # trot_mat = Operator(qc_trot).data    
    # print('scaling=', parameters['scaling'],'E_L=',E_L, 'g=', parameters['g'])

    # print('\n2-norm difference:', np.linalg.norm(trot_mat-exact_mat, ord=2))
    # trot_e, trot_v = eig(trot_mat)
    # exact_e, exact_v = eig(exact_mat)
    # trot_phase = np.sort(-np.log(trot_e).imag)
    # exact_phase = np.sort(-np.log(exact_e).imag)
    # print('Trotter Spectrum Phases:', trot_phase)
    # print('Exact Spectrum Phases:', exact_phase)


    # U_trans=U
    # U=UnitaryGate(trot_mat)

    
    # trans_mat = Operator(trans_qc).data
    # trans_mat=U_trans.to_matrix()

    # print('Absolute error: ', abs(trot_mat - trans_mat))
    # frob_norm = np.linalg.norm(trot_mat - trans_mat, ord=2)
    # print('Frobenius norm difference: ', frob_norm)
    # trot_e, trot_v = eig(trot_mat)
    # trans_e, trans_v = eig(trans_mat)
    # trot_phase = np.sort(-np.log(trot_e).imag)
    # trans_phase = np.sort(-np.log(trans_e).imag)

    # print('Trotter Spectrum Phases', trot_phase)
    # print('Qiskit Spectrum Phases', trans_phase)
    # print("Phases norm Difference", np.linalg.norm(trot_phase-trans_phase))
    # print('Ground state overlap with qiskit trans: ', abs(ground_state.conj().T@trans_mat@ground_state)**2)
    # print('Ground state overlap with trotter trans: ', abs(ground_state.conj().T@trot_mat@ground_state)**2)

    #---+---#

    # print('(Trotter matrix).conj().T@(Qiskit Matrix): \n', trot_mat.conj().T@trans_mat)

    # trans_qc = create_trot_ht(parameters, backend, qc_trot_unit, modified=True)
    # trans_qc.draw(output='mpl', filename='HT_site2_trot_trans_circ.pdf')

    # trans_qcs.append(trans_qc)
    # trans_qc = create_trot_ht(parameters, backend, qc_trot_unit, W='Im', modified=True)

    # print('Imaginary modified Hadamard test gate counts:', trans_qc.count_ops())
    # trans_qcs.append(trans_qc)
    
    # results = sampler.run(trans_qcs, shots = shots).result()
    
    # raw_data = results[0].data
    # cbit = list(raw_data.keys())[0]
    # counts = raw_data[cbit].get_counts()
    # Re = calculate_exp_vals(counts, shots)
    # raw_data = results[1].data
    # cbit = list(raw_data.keys())[0]
    # counts = raw_data[cbit].get_counts()
    # Im = calculate_exp_vals(counts, shots)

    # print('Real:', eig_val[0])
    # print('Estimate:', -np.log(complex(Re, Im)).imag)