import numpy as np
pi = np.pi 
from Service import create_hardware_backend
import matplotlib.pyplot as plt

from scipy.linalg import expm, eigh, norm, eig

from qiskit import transpile
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate, StatePreparation

from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler, EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService as QRS
from qiskit.quantum_info import Operator, Statevector 
from matplotlib.colors import LogNorm

def rayleigh(M, x):
    return ((x.conj().T @ M @ x) / (x.conj().T @ x))[0][0] # turn it into a scalar

# def lowest_energy(H):
#     do:

#     while() 

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
    qubits = parameters['qubits']
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
    qubits = parameters['qubits']
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

def get_hubb_ham(L, t, V):
    n_qubits = 2 * L
    terms = []
    Loc_V = [("II", 1/4), ("ZI", -1/4), ("IZ", -1/4), ("ZZ", 1/4)]
    XY = ["X", "Y"]

    for j in range(L):
        # Interaction terms (V*n_up*n_down)
        q_up, q_dn = 2*j, 2*j + 1
        # Local V terms (V/4*(I - Z_up - Z_dn + Z_up*Z_dn))
        for pauli, coeff in Loc_V:
            s = ["I"] * n_qubits
            s[q_up], s[q_dn] = pauli[0], pauli[1]
            terms.append(("".join(s), V*coeff))

        # Hopping terms (-t*(c's...))
        # Hop from SITE j to j+1 (open bconds)

        if j < L - 1:
            # For spin-up qubit 2*j to qubit 2*j+2. Jumps over qubit (2*j+1)
            # For spin-down qubit 2*j+1 to qubit 2*j+3. Jumps over qubit (2*j+2)
            for spin_offset in [0, 1]: # 0: up, 1: down
                q1, q2 = 2*j + spin_offset, 2*(j+1) + spin_offset
                for op in XY:
                    s = ["I"] * n_qubits
                    s[q1], s[q2] = op, op
                    terms.append(("".join(s), -t/2))

    # print(terms)
    return SparsePauliOp.from_list(terms)

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
    sys = parameters['sys'][0:4].upper()
    if 'qubits' in parameters.keys(): qubits = parameters['qubits']
    else: qubits = 2
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    if sys=="TFIM":
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
    elif sys == 'HEIS':
        # jordan-wigner form
        H_int = [["I"] * qubits for _ in range(3 * (qubits - 1))]
        for i in range(qubits - 1):
            H_int[i][i] = "Z"
            H_int[i][i + 1] = "Z"
        for i in range(qubits - 1):
            H_int[qubits - 1 + i][i] = "X"
            H_int[qubits - 1 + i][i + 1] = "X"
        for i in range(qubits - 1):
            H_int[2 * (qubits - 1) + i][i] = "Y"
            H_int[2 * (qubits - 1) + i][i + 1] = "Y"
        H_int = ["".join(term) for term in H_int]
        H_tot = [(term, 1) if term.count("Z") == 2 else (term, 1) for term in H_int]

        H_op = SparsePauliOp.from_list(H_tot)
        H = H_op.to_matrix()
    elif sys == 'HUBB':
        assert(parameters['qubits']%2==0)
        L = parameters['qubits']//2 # L is sites, N = 2L is qubits
        T = parameters['T']
        V = parameters['V']
        H_op = get_hubb_ham(L,T,V)

        # J_x = -T/2 # flip-flop strength for hubb
        # J_z = V/4 # antisotropy coupling strength for hubb
        # B = -V/2 # Z term strength

        # # jordan-wigner form
        # terms = []
        # for i in range(qubits - 1):
        #     terms.append(( "I"*i + "XX" + "I"*(qubits - i - 2), J_x ))
        #     terms.append(( "I"*i + "YY" + "I"*(qubits - i - 2), J_x ))

        #     terms.append(( "I"*i + "ZZ" + "I"*(qubits - i - 2), J_z ))

        # for i in range(qubits):
        #     terms.append(( "I"*i + "Z" + "I"*(qubits - i - 1), -B ))

        # H_op = SparsePauliOp.from_list(terms)
        H = H_op.to_matrix()
    else: assert(False)


    val, vec = eigh(H)
    real_E_0 = val[0]
    # power method

    largest_eigenvalue = np.max(abs(val)) # use lambda_new when the above code segment
    unscaled_H = np.copy(H)
    if scale:
        # scale eigenvalues of the Hamiltonian
        n = 2**qubits
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
    return unscaled_H, H, real_E_0, largest_eigenvalue

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

def tfim_trotter_evolution(j, g, n_qubits, t=1.0, num_time_steps=1):
    j=-j
    g=-g
    qc = QuantumCircuit(n_qubits)
    dt = t / num_time_steps
    for _ in range(num_time_steps):
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

def spin_chain_trotter_evolution(n_qubits, t=1, num_time_steps=1, trotter_order=1):
    dt=t/num_time_steps
    # Create instruction for rotation about XX+YY-ZZ:
    Rxyz_circ = QuantumCircuit(2)
    if trotter_order==1:
        Rxyz_circ.rxx(dt, 0, 1)
        Rxyz_circ.ryy(dt, 0, 1)
        Rxyz_circ.rzz(dt, 0, 1)
    if trotter_order==2:
        Rxyz_circ.rxx(dt/2, 0, 1)
        Rxyz_circ.rzz(dt/2, 0, 1)
        Rxyz_circ.ryy(dt, 0, 1)
        Rxyz_circ.rzz(dt/2, 0, 1)
        Rxyz_circ.rxx(dt/2, 0, 1)
    Rxyz_instr = Rxyz_circ.to_instruction(label="RXX+YY+ZZ")

    interaction_list = [
        [[i, i + 1] for i in range(0, n_qubits - 1, 2)],
        [[i, i + 1] for i in range(1, n_qubits - 1, 2)],
    ]  # linear chain

    qr = QuantumRegister(n_qubits)
    trotter_step_circ = QuantumCircuit(qr)
    for i, color in enumerate(interaction_list):
        for interaction in color:
            trotter_step_circ.append(Rxyz_instr, interaction)
        if i < len(interaction_list) - 1:
            trotter_step_circ.barrier()
    reverse_trotter_step_circ = trotter_step_circ.reverse_ops()
    
    qc_evol = QuantumCircuit(qr)
    for step in range(num_time_steps):
        if step % 2 == 0:
            qc_evol = qc_evol.compose(trotter_step_circ)
        else:
            qc_evol = qc_evol.compose(reverse_trotter_step_circ)
    # qc_evol.decompose().draw("mpl", fold=-1, scale=0.5)
    return qc_evol

def hubbard_trotter_evolution(n_qubits, J_x, J_z, B, final_T=1, num_time_steps=1):
    # trotterized circuit for hubbard model
    times = np.linspace(0, final_T, num_time_steps)

    xt = J_x*times
    zt = J_z*times
    Bt = B*times

    # Create instruction for rotation about XX+YY-ZZ:
    instrs = []
    for i in range(num_time_steps):
        Rxyz_circ = QuantumCircuit(2)
        Rxyz_circ.rxx(xt[i], 0, 1)
        Rxyz_circ.ryy(xt[i], 0, 1)
        Rxyz_circ.rzz(zt[i], 0, 1)
        Rxyz_instr = Rxyz_circ.to_instruction(label="RXX+YY+ZZ")
        instrs.append(Rxyz_instr)
    interaction_list = [
        [[i, i + 1] for i in range(0, n_qubits - 1, 2)],
        [[i, i + 1] for i in range(1, n_qubits - 1, 2)],
    ]  # linear chain
    t_stp_circs = []
    r_t_stp_circs = []
    for j in range(num_time_steps):
        qr = QuantumRegister(n_qubits)
        trotter_step_circ = QuantumCircuit(qr)
        for i, color in enumerate(interaction_list):
            for interaction in color:
                trotter_step_circ.append(instrs[j], interaction)
            if i < len(interaction_list) - 1:
                trotter_step_circ.barrier()
        for i in range(n_qubits):
            trotter_step_circ.rz(Bt[j], i)
        reverse_trotter_step_circ = trotter_step_circ.reverse_ops()

        t_stp_circs.append(trotter_step_circ)
        r_t_stp_circs.append(reverse_trotter_step_circ)

    qc_evol = QuantumCircuit(qr)
    for step in range(num_time_steps):
        if step % 2 == 0:
            qc_evol = qc_evol.compose(t_stp_circs[step])
        else:
            qc_evol = qc_evol.compose(r_t_stp_circs[step])

    # qc_evol.decompose().draw("mpl")
    return qc_evol

# params scaled beforehand
def trotter_evolution(t, parameters):
    sys = parameters['sys'][0:4].upper()
    sites = parameters['qubits']
    num_time_steps = parameters['num_time_steps']
    
    if sys=='TFIM':
        J = parameters['J']
        g = parameters['g']
        qc = tfim_trotter_evolution(J, g, sites, t=-t, num_time_steps=num_time_steps)
    elif sys=='HEIS':
        if 'trotter_order' in parameters:
            trotter_order = parameters['trotter_order']
        else:
            trotter_order = 1
        qc = spin_chain_trotter_evolution(sites, t=t, num_time_steps=num_time_steps, trotter_order=trotter_order)
    elif sys == 'HUBB':
        T = parameters['T']
        V = parameters['V']

        J_x = -T/2 # flip-flop strength for hubb
        J_z = V/4 # antisotropy coupling strength for hubb
        B = -V/2 # Z term strength

        qc = hubbard_trotter_evolution(sites, J_x, J_z, B, final_T=t, num_time_steps=num_time_steps)
    return qc

def r(mat, theta):
    return expm(-1j*theta/2*mat)

def check_matrices(A, B, precision, disp=True):
    if not norm(A-B) < precision:
        if disp:
            print('Failed Check')
            print('A', A)
            print('B', B)
            print('norm', norm(A-B))
        return False
    return True

def get_spectrum(A):
    if check_matrices(A, A.conj().T, 1E-12, disp=False):
        eigval_A,_ = eigh(A)
    else:
        eigval_A,_ = eig(A)
    return np.sort([-np.log(eigval).imag for eigval in eigval_A])


def check_spectrum(A, B, precision):
    phases_A = get_spectrum(A)
    phases_B = get_spectrum(B)
    diff = norm(phases_A-phases_B)
    if  diff < precision:
        return True
    else:
        print('phases:')
        print(phases_A)
        print(phases_B)
        print('diff =', diff)


def exact_trotter_matrix_TFIM(sites, mag_field_mat, g, coupling_mat, J, scaled_t, trotter_steps):
    full_mat = np.eye(2**sites, dtype=complex)
    dt = scaled_t / trotter_steps
    for _ in range(trotter_steps):
        beta_half = g * (dt/2)
        alpha = J * dt
        part1 = 1
        for _ in range(sites):
            part1 = np.kron(part1, r(mag_field_mat, 2*beta_half))
        part2 = np.eye(2**sites, dtype=complex)
        for qubit in range(sites-1):
            temp = 1
            for index in range(sites-1):
                if index == qubit:
                    mat = r(np.kron(coupling_mat, coupling_mat), 2*alpha)
                else:
                    mat = np.eye(2, dtype=complex)
                temp = np.kron(temp, mat)
            part2 @= temp
        part3 = 1
        for _ in range(sites):
            part3 = np.kron(part3, r(mag_field_mat, 2*beta_half))
        trotter_step_mat = part1 @ part2 @ part3
        full_mat @= trotter_step_mat
    return full_mat

def exact_trotter_matrix_spin(sites, coupling_mats, scaled_t, time_steps, trotter_order=1):
    full_mat = np.eye(2**sites, dtype=complex)
    dt = scaled_t / time_steps

    Rxyz_mat = np.eye(4, dtype=complex)
    if trotter_order==1:
        for mat in coupling_mats:
            Rxyz_mat @= r(np.kron(mat,mat), dt)
    elif trotter_order==2:
        # loop to have symmetric ordering
        for index in range(len(coupling_mats)):
            mat = coupling_mats[index]
            if index != len(coupling_mats)-1:
                Rxyz_mat @= r(np.kron(mat,mat), dt/2)
            else:
                Rxyz_mat @= r(np.kron(mat,mat), dt) # inflection
        for i in range(len(coupling_mats)-1):
            index = len(coupling_mats)-2-i
            mat = coupling_mats[index]
            Rxyz_mat @= r(np.kron(mat,mat), dt/2)
    
    interaction_list = [
        [[i, i + 1] for i in range(0, sites - 1, 2)],
        [[i, i + 1] for i in range(1, sites - 1, 2)],
    ]  # linear chain
    
    forward_trotter_step_mat = np.eye(2**sites, dtype=complex)
    reverse_trotter_step_mat = np.eye(2**sites, dtype=complex)
    for _, color in enumerate(interaction_list):
        for interaction in color:
            pair_mat = 1
            for site in range(sites-1):
                if site == interaction[0]:
                    pair_mat = np.kron(pair_mat, Rxyz_mat)
                else:
                    pair_mat = np.kron(pair_mat, np.eye(2))
            forward_trotter_step_mat @= pair_mat
            reverse_trotter_step_mat = pair_mat@reverse_trotter_step_mat
    full_mat = np.eye(2**sites, dtype=complex)
    for step in range(time_steps):
        if step % 2 == 0:
            full_mat @= forward_trotter_step_mat
        else:
            full_mat @= reverse_trotter_step_mat
    return full_mat

def run_tests():
    # t = .0017 # first order max T to get 10^-3
    t = .0017
    trotter_order = 2
    parameters = {}
    parameters['qubits']    = 2
    parameters['scaling']  = 1
    parameters['shifting'] = 0

    # create_hamiltonian_tests
    H_datas = {}
    parameters['sys']      = "TFIM"
    parameters['g']        = 4
    parameters['J']        = 1 
    H, scaled_H, E_0, E_L = create_hamiltonian(parameters, scale=True, show_steps=False)
    scaled_t = t * parameters['scaling']/E_L
    U_t = expm(-1j*H*scaled_t)
    U_H = expm(-1j*scaled_H*t)
    assert(check_matrices(U_t, U_H, 1E-8))
    H_datas['TFIM'] = [np.copy(U_t), E_0, E_L, scaled_t]

    parameters['sys']      = "HEIS"
    H, scaled_H, E_0, E_L = create_hamiltonian(parameters, scale=True, show_steps=False)
    scaled_t = t * parameters['scaling']/E_L
    U_t = expm(-1j*H*scaled_t)
    U_H = expm(-1j*scaled_H*t)
    assert(check_matrices(U_t, U_H, 1E-8))
    H_datas['HEIS'] = [np.copy(U_t), E_0, E_L, scaled_t]

    parameters['sys']      = "HUBB"
    parameters['L']        = 4
    parameters['T']        = 1
    parameters['V']        = 1.5
    H, scaled_H, E_0, E_L = create_hamiltonian(parameters, scale=True, show_steps=False)
    scaled_t = t * parameters['scaling']/E_L
    U_t = expm(-1j*H*scaled_t)
    U_H = expm(-1j*scaled_H*t)
    assert(check_matrices(U_t, U_H, 1E-8))
    H_datas['HUBB'] = [np.copy(U_t), E_0, E_L, scaled_t]

    X = np.array([[0, 1],
                  [1, 0]])
    Y = np.array([[0, -1j],
                  [1j,  0]])
    Z = np.array([[1,  0],
                  [0, -1]])

    # tfim_trotter_evolution
    print("STARTING TFIM TESTS")
    scaled_t = H_datas['TFIM'][3]
    
    trotter_circ = tfim_trotter_evolution(parameters['J'], parameters['g'], parameters['qubits'], t=scaled_t, num_time_steps=2)
    trotter_circ_mat = Operator(trotter_circ).data

    # circuit correctness check
    print("\tCORRECTNESS CHECK")
    exact_trotter_mat = exact_trotter_matrix_TFIM(parameters['qubits'], X, parameters['g'], Z, parameters['J'], -scaled_t, 2)
    assert(check_matrices(trotter_circ_mat, exact_trotter_mat, 1E-12))
    assert(check_spectrum(trotter_circ_mat, exact_trotter_mat, 1E-12))
    print("\tCORRECTNESS CHECK COMPLETED")


    # check precision
    print("\tPRECISION CHECK")
    assert(check_matrices(trotter_circ_mat, H_datas['TFIM'][0], 1E-2))
    assert(check_spectrum(trotter_circ_mat, H_datas['TFIM'][0], 1E-2))

    trotter_circ = tfim_trotter_evolution(parameters['J'], parameters['g'], parameters['qubits'], t=scaled_t, num_time_steps=20)
    trotter_circ_mat = Operator(trotter_circ).data
    assert(check_matrices(trotter_circ_mat, H_datas['TFIM'][0], 1E-4))
    assert(check_spectrum(trotter_circ_mat, H_datas['TFIM'][0], 1E-4))

    trotter_circ = tfim_trotter_evolution(parameters['J'], parameters['g'], parameters['qubits'], t=scaled_t, num_time_steps=2000)
    trotter_circ_mat = Operator(trotter_circ).data
    assert(check_matrices(trotter_circ_mat, H_datas['TFIM'][0], 1E-8))
    assert(check_spectrum(trotter_circ_mat, H_datas['TFIM'][0], 1E-8))

    print("COMPLETED TFIM TESTS")

    # heis_trotter_evolution
    print("STARTING SPIN CHAIN TESTS")
    parameters['sys']      = "HEIS"
    scaled_t = H_datas['HEIS'][3]

    trotter_circ = spin_chain_trotter_evolution(parameters['qubits'], t=scaled_t, num_time_steps=1)
    trotter_circ_mat = Operator(trotter_circ).data

    # circuit correctness check
    print("\tCORRECTNESS CHECK")
    exact_trotter_mat = exact_trotter_matrix_spin(parameters['qubits'], [X,Z,Y], scaled_t, 1, trotter_order=trotter_order)
    assert(check_matrices(trotter_circ_mat, exact_trotter_mat, 1E-12))
    assert(check_spectrum(trotter_circ_mat, exact_trotter_mat, 1E-12))

    # precision check
    print("\tPRECISION CHECK")
    trotter_circ = spin_chain_trotter_evolution(parameters['qubits'], t=scaled_t, num_time_steps=1, trotter_order=trotter_order)
    trotter_circ_mat = Operator(trotter_circ).data
    assert(check_matrices(trotter_circ_mat, H_datas['HEIS'][0], 1E-3))
    assert(check_spectrum(trotter_circ_mat, H_datas['HEIS'][0], 1E-3))

    print("COMPLETE SPIN CHAIN TESTS")

    # hubb_trotter_evolution
    print("STARTING HUBBARD TESTS")
    parameters['sys']      = "HUBB"
    scaled_t = H_datas['HUBB'][3]

    T = parameters['T']
    V = parameters['V']

    J_x = -T/2 # flip-flop strength for hubb
    J_z = V/4 # antisotropy coupling strength for hubb
    B = -V/2 # Z term strength

    trotter_circ = hubbard_trotter_evolution(parameters['qubits'], J_x, J_z, B, final_T=scaled_t, num_time_steps=2)
    trotter_circ_mat = Operator(trotter_circ).data
    assert(check_matrices(trotter_circ_mat, H_datas['HUBB'][0], 1E-2))
    assert(check_spectrum(trotter_circ_mat, H_datas['HUBB'][0], 1E-2))
    print("COMPLETED HUBBARD TESTS")
import scipy as sp, itertools as it
def single_particle_gs(H_op, n_qubits):
    """
    Find the ground state of the single particle(excitation) sector
    """
    H_x = []
    for p, coeff in H_op.to_list():
        H_x.append(set([i for i, v in enumerate(Pauli(p).x) if v]))

    H_z = []
    for p, coeff in H_op.to_list():
        H_z.append(set([i for i, v in enumerate(Pauli(p).z) if v]))

    H_c = H_op.coeffs

    # print("n_sys_qubits", n_qubits)

    n_exc = 1
    sub_dimn = int(sp.special.comb(n_qubits + 1, n_exc))
    # print("n_exc", n_exc, ", subspace dimension", sub_dimn)

    few_particle_H = np.zeros((sub_dimn, sub_dimn), dtype=complex)

    sparse_vecs = [
        set(vec) for vec in it.combinations(range(n_qubits + 1), r=n_exc)
    ]  # list all of the possible sets of n_exc indices of 1s in n_exc-particle states

    m = 0
    for i, i_set in enumerate(sparse_vecs):
        for j, j_set in enumerate(sparse_vecs):
            m += 1

            if len(i_set.symmetric_difference(j_set)) <= 2:
                for p_x, p_z, coeff in zip(H_x, H_z, H_c):
                    if i_set.symmetric_difference(j_set) == p_x:
                        sgn = ((-1j) ** len(p_x.intersection(p_z))) * (
                            (-1) ** len(i_set.intersection(p_z))
                        )
                    else:
                        sgn = 0

                    few_particle_H[i, j] += sgn * coeff

    gs_en = min(np.linalg.eigvalsh(few_particle_H))
    # print("single particle ground state energy: ", gs_en)
    return gs_en

def get_total_number_operator(L):
    n_qubits = 2 * L
    dim = 2**n_qubits
    
    # Standard building blocks
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    n_base = 0.5 * (I - Z) # The single-qubit number operator

    def get_single_n(index):
        op_list = []
        for i in range(n_qubits):
            if i == index: op_list.append(n_base)
            else:          op_list.append(I)
        
        res = op_list[0]
        for next_op in op_list[1:]:
            res = np.kron(res, next_op)
        return res

    N_total = np.zeros((dim, dim))
    for i in range(n_qubits):
        N_total += get_single_n(i)
        
    return N_total

def rayleigh_vs_t(qubits):
    # create example circuit
    parameters = {}
    parameters['sys']      = "HUBB"
    parameters['qubits']   = qubits
    parameters['scaling']  = 1
    parameters['shifting'] = 0
    #HUBB
    parameters['T']        = 1
    parameters['V']        = 1.5

    unscaled_H, H, E_0, E_L = create_hamiltonian(parameters, scale=True)
    E_real = get_ground_hubb(parameters)
    print('real lowest energy:', E_real)
    precision = 1E-12

    # N = get_total_number_operator(parameters['qubits']//2)
    # unscaled_H -= precision*N
    # eig_val, eig_vec = eig(unscaled_H)
    # init = [[i] for i in eig_vec[:,3]]

    one = [[0],[1]]
    zero = [[1],[0]]
    init = [1]
    for i in range(parameters['qubits']):
        if i == parameters['qubits'] // 2-1:
            init = np.kron(one, init)
        else:
            init = np.kron(zero, init)

    # print(init)
    # print(init)
    # print(init)
    diff = np.inf
    t = 0
    i = 0
    E_test_list = []
    t_list = []
    while diff > precision*10 and i < 1000:
        # for site in sites
        
        qc_trot_unitary = get_hubb(t, qubits, 1, parameters['T'], parameters['V'])
        time_evol = Operator(qc_trot_unitary).data
        # time_evol = expm(-1.0j*unscaled_H*t) 

        sv = time_evol @ init
        print(sv)
        E_test = rayleigh(unscaled_H, sv)
        
        t_list.append(t)
        diff = abs(E_test-E_real)
        E_test_list.append(diff)
        # print(E_test)
        i += 1
        t += 1

    
    plt.plot(t_list, E_test_list, label = 'Energy Difference')
    plt.plot(t_list, [1E-3] * len(t_list), label='chemical accuracy')
    # plt.yscale('log')
    plt.title('Rayleigh vs t')
    plt.xlabel('time')
    plt.ylabel('Energy Difference')
    plt.legend()
    plt.savefig('testing_graphs/rayleigh_vs_t.pdf')
    return t_list[-1]
    

def get_ground_hubb(parameters):
    L = parameters['qubits']//2
    T = parameters['T']
    V = parameters['V']
    pauli_string = get_hubb_ham(L, T, V)
    return single_particle_gs(pauli_string, parameters['qubits'])

def trotter_order_vs_sites(parameters = {}):

    max_t = 2
    t_div = 1000
    t_list = [max_t*(i+1)/t_div for i in range(t_div)]
    trotter_order_list = [1,2]
    for trotter_order in trotter_order_list:
        parameters['trotter_order'] = trotter_order
        trotter_diff = []
        sites_list = [2,3,4,5]#np.arange(2,7,1)
        for sites in sites_list:

            print("Sites:", sites)

            parameters['scaling']  = 1
            parameters['shifting'] = 0
            parameters['qubits']    = sites

            t_diffs = []
            for t in t_list:

                H, scaled_H, E_0, E_L = create_hamiltonian(parameters, scale=True)
                energy, eig_vec = eig(H)
                ground_state = eig_vec[:,0]

                # print("hamiltonian energy", energy[0:4])

                U = expm(-1j*scaled_H*t)
                scaling_coeff = E_L/parameters['scaling']
                # time is scaled down by 1/2?
                scaled_t = t/scaling_coeff

                if parameters['sys'] == "HEIS":
                    scaled_t *= 2
                
                # try to use trotter_evolution instead
                trotter_circ = trotter_evolution(scaled_t, parameters)
                
                trotter_circ_mat = Operator(trotter_circ).data
                circ_spec = [i/t*scaling_coeff for i in get_spectrum(trotter_circ_mat)]
                exact_spec = [i/t*scaling_coeff for i in get_spectrum(U)]
                
                # energy check

                # print("\tExact spectrum", exact_spec[0:4])
                # print("\tCircuit spectrum", circ_spec[0:4])
                # diff_list = abs(np.array(circ_spec) - np.array(exact_spec))
                # print("\tdiffs", diff_list[0:4])
                # print("\tReal Energy", E_0)
                assert(abs(exact_spec[0] - E_0)<1E-11)

                t_diff = abs(circ_spec[0] - exact_spec[0])
                t_diffs.append(t_diff)
            plt.plot(t_list, t_diffs, label="order="+str(trotter_order))
        plt.title("T vs Error for "+str(sites)+" sites")
        plt.ylabel("Error (abs error spectrum)")
        plt.xlabel("Time (s)")
        # plt.yscale("log")
        plt.savefig("testing_graphs/t_vs_error_w_sites="+str(sites)+".pdf")
        plt.close()
    return

def get_hubbard_instrs(dt, T, V):
    J_x = -T/2
    B = V/4
    # Hopping (XX+YY)
    h_circ = QuantumCircuit(2)
    h_circ.rxx(J_x*dt, 0, 1)
    h_circ.ryy(J_x*dt, 0, 1)
    
    # Interaction (ZZ + Z + Z)
    i_circ = QuantumCircuit(2)
    i_circ.rz(-B*dt, 0)
    i_circ.rz(-B*dt, 1)
    i_circ.rzz(B*dt, 0, 1)
    
    return h_circ.to_instruction(label=f"Hop({dt})"), i_circ.to_instruction(label=f"Int({dt})")

def get_hubb(t, n_qubits, num_trotter_steps, T, V):# First Order Trotterization

    hop_instr, int_instr = get_hubbard_instrs(t, T, V)
    qr = QuantumRegister(n_qubits)
    qc_evol = QuantumCircuit(qr)

    for step in range(num_trotter_steps):        
        # Hopping
        # Even
        for i in range(0, n_qubits - 2, 2):
            qc_evol.append(hop_instr, [qr[i], qr[i+2]])
        # Odd
        for i in range(1, n_qubits - 2, 2):
            qc_evol.append(hop_instr, [qr[i], qr[i+2]])
        qc_evol.barrier()

        # Interaction
        for i in range(0, n_qubits, 2):
            qc_evol.append(int_instr, [qr[i], qr[i+1]])
        
    # qc_evol.decompose().draw("mpl")
    # qc_evol.draw("mpl")
    return qc_evol

def heatmap():
    use_hardware = False
    if use_hardware:
        backend = create_hardware_backend()
    else:
        backend = AerSimulator()
    sampler = Sampler(backend)

    t = 1
    sites_list = np.arange(2,10,2)
    trot_list = np.arange(2,11,1).tolist()

    print('sites list', sites_list)
    print('trot list', trot_list)

    norm_diffs = []
    depths_2q = []
    # qiskit_depths = []
    spectral_diffs = []
    spectral_sorted_diffs = []
    for i in range(len(sites_list)):
        norm_diffs.append([])
        spectral_diffs.append([])
        spectral_sorted_diffs.append([])

        depths_2q.append([])
        print('Generating data for',sites_list[i],'sites')

        # create example circuit
        parameters = {}
        parameters['sys']      = "TFIM"
        parameters['qubits']    = sites_list[i]
        parameters['scaling']  = 1
        parameters['shifting'] = 0
        #TFIM
        parameters['g']        = 4
        parameters['J']        = 1 
        #HUBB
        parameters['T']        = 1
        parameters['V']        = 1

        unscaled_H, H, E_0, E_L = create_hamiltonian(parameters, scale=True)
        scaled_t = t*(parameters['scaling']/E_L)
        print(scaled_t)

        U1 = expm(-1j*unscaled_H*scaled_t)
        U = expm(-1j*H*t)
        print(H)
        assert(norm(U1 - U) < 1E-8) 
        if norm(U-U.T) < 1E-8:
            U_eigval, _ = eigh(U)
        else:
            U_eigval, _ = eig(U)
        U_phases = [-np.log(eigval).imag for eigval in U_eigval]
        U_phases_sorted = np.sort(U_phases)
        
        for j in range(len(trot_list)):
            print('  trotter steps:', trot_list[j])
            parameters['num_time_steps'] = trot_list[j]
            J_x = -parameters['T']/2 # flip-flop strength for hubb
            J_z = parameters['V']/4 # antisotropy coupling strength for hubb
            B = -parameters['V']/2 # Z term strength
            qc_trot_unitary = get_hubb(sites_list[i], trot_list[j])
            # qc_trot_unitary = trotter_evolution(-scaled_t, parameters)
            check = Operator(qc_trot_unitary).data
            unscaled_time_test = expm(-1j*H*t)
            print(check, "\n")
            print(unscaled_time_test)
            # assert(norm(check - unscaled_time_test) < 1E-8)
            scaled_time_test = expm(-1j*unscaled_H*scaled_t)
            # print(test)
            # assert(norm(check - scaled_time_test) < 1E-8)
            # print(qc_trot_unitary)
            
            trans_qc_trot = qc_trot_unitary
            
            trans_qc_trot = transpile(qc_trot_unitary, optimization_level=3)#, basis_gates=['id','ecr','rz','sx','x']) # just rpi_rensselaer basis gates
            trot_mat = Operator(trans_qc_trot).data
            
            norm_diffs[i].append(np.linalg.norm(trot_mat-U, ord=2))

            trot_eigval, _ = eig(trot_mat)
            trot_phases = [-np.log(eigval).imag for eigval in trot_eigval]

            trot_phases_sorted = np.sort(trot_phases)
            small_diff = abs(U_phases_sorted[0]-trot_phases_sorted[0])
            sum = 0
            for phase_index in range(len(U_phases_sorted)):
                sum += abs(trot_phases_sorted[phase_index]-U_phases_sorted[phase_index])
            spectral_sorted_diffs[i].append(sum)
            
            # qc_trot = create_trot_ht(parameters, backend, qc_trot_unitary, modified=True)
            # qiskit_mat = Operator(qc_qiskit).data
            # trans_qc_trot = transpile(qc_trot, optimization_level=3, basis_gates=['id','ecr','rz','sx','x']) # just rpi_rensselaer basis gates
            # print(' gate counts:', trans_qc_trot.count_ops())
            gate_count = trans_qc_trot.num_nonlocal_gates()
            depths_2q[i].append(gate_count)
        
    # print(norm_diffs)
    # print(spectral_diffs)
    # print(depths_2q)
    # for i in range(len(depths_2q)): depths_2q[i].append(qiskit_depths[i])

    size = 1
    fig, axes = plt.subplots(1, size, figsize=(2*len(trot_list),len(sites_list)))
    fig.suptitle(parameters['sys'])
    if size==1: axes = [axes]
    index = 0

    # colorbar_scale = LogNorm(vmin=10**0, vmax=10**-5)
    # axe = axes[index]
    # im1 = axe.imshow(norm_diffs, norm=colorbar_scale)#LogNorm(vmin=np.min(norm_diffs), vmax=np.max(norm_diffs)))
    # for ax in axes:
    #     axe.set_xticks(range(len(trot_list)), labels=np.array(trot_list).astype(str),
    #                 rotation=45, ha="right", rotation_mode="anchor")
    #     axe.set_yticks(range(len(sites_list)), labels=np.array(sites_list).astype(str))
    # index+=1
    # cbar = fig.colorbar(im1, ax=axes[0])
    # cbar.set_label('2-norm difference')

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(sites_list)):
    #     for j in range(len(trot_list)):
    #         axes[0].text(j, i, f'{np.log(norm_diffs[i][j]):1.2f}', ha="center", va="center", color="w")

    # axe = axes[index]
    # spectrum_sorted_diff_ax = axe.imshow(spectral_sorted_diffs, norm=colorbar_scale)#LogNorm(vmin=np.min(spectral_sorted_diffs), vmax=np.max(spectral_sorted_diffs)))
    # cbar = fig.colorbar(spectrum_sorted_diff_ax, ax=axe)
    # cbar.set_label('Phase difference for sorted spectrum')
    # index += 1

    axe = axes[index]
    axe.set_xticks(range(len(trot_list)), labels=np.array(trot_list).astype(str),
                rotation=45, ha="right", rotation_mode="anchor")
    axe.set_yticks(range(len(sites_list)), labels=np.array(sites_list).astype(str))
    # depth graph
    # axe.set_title("Trotterized HUBB TE U err")
    # axe.set_xlabel('Trotter Steps')
    # axe.set_ylabel('Sites')
    im2 = axe.imshow(depths_2q)
    # Show all ticks and label them with the respective list entries
    # trot_list.append('qiskit')
    axe.set_xticks(range(len(trot_list)), labels=np.array(trot_list).astype(str),
                rotation=45, ha="right", rotation_mode="anchor")
    axe.set_yticks(range(len(sites_list)), labels=np.array(sites_list).astype(str))
    cbar = fig.colorbar(im2, ax=axe)
    cbar.set_label('2-qubit gate counts')
    axe.set_title("Modified HT HUBB 2-q gate counts")
    axe.set_xlabel('Trotter Steps/Method')
    axe.set_ylabel('Sites')
    # Loop over data dimensions and create text annotations.
    for i in range(len(sites_list)):
        for j in range(len(trot_list)):
            text = axe.text(j, i, depths_2q[i][j],
                        ha="center", va="center", color="w")
    index += 1
    
    plt.savefig('testing_graphs/heatmap.pdf')
    plt.close()


    # qc_trot = trotter_evolution(-trot_j, -trot_g, parameters['qubits'], t=t, r=r)   # increase r to reduce Trotter error
    # qc_trot.draw(output='mpl', filename='HT_site'+str(parameters['qubits'])+'_trot.pdf')
    # qc_trot = transpile(qc_trot, optimization_level=3, basis_gates=['id','ecr','rz','sx','x']) # just rpi_rensselaer basis gates
    # # qc_trot = transpile(qc_trot, backend, optimization_level=3) # rpi_rensselaer basis gates + qpu mapping gates (swapping, etc.)
    # qc_trot.draw(output='mpl', idle_wires=False, filename='HT_opt3_site'+str(parameters['qubits'])+'_trot.pdf')
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

if __name__ == '__main__':
    # np.set_printoptions(linewidth=300, suppress=True)
    # run_tests()
    # trotter_order_vs_sites(parameters={"sys":"HEIS", "trotter_order":1, "num_time_steps":1})
    # exit(1)
    final_T = rayleigh_vs_t(4)
    print('Final Time:', final_T)