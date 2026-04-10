import numpy as np
import itertools as it
import scipy as sp
import matplotlib.pyplot as plt
from scipy.linalg import eig, expm
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator

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

def get_ground_hubb(parameters):
    L = parameters['qubits']//2
    T = parameters['T']
    V = parameters['V']
    pauli_string = get_hubb_ham(L, T, V)
    return single_particle_gs(pauli_string, parameters['qubits'])

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
    sys = parameters['sys'][0:4].upper()
    if 'qubits' in parameters.keys(): qubits = parameters['qubits']
    else: qubits = 2
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    if sys == 'HUBB':
        assert(parameters['qubits']%2==0)
        L = parameters['qubits']//2 # L is sites, N = 2L is qubits
        T = parameters['T']
        V = parameters['V']
        H_op = get_hubb_ham(L,T,V)
        H = H_op.to_matrix()
    else: assert(False)
    return H

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

def get_hubb(dt, n_qubits, T, V):# First Order Trotterization
    hop_instr, int_instr = get_hubbard_instrs(dt, T, V)
    qr = QuantumRegister(n_qubits)
    qc_evol = QuantumCircuit(qr)
       
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


#--------------------------------------------#


parameters = {}
parameters['sys']      = "HUBB"
parameters['qubits']   = 4 # 2*sites
#HUBB
parameters['T']        = 1
parameters['V']        = 1.5

H = create_hamiltonian(parameters, scale=True)
H_eigs, _ = eig(H)
E_real = get_ground_hubb(parameters)
print('real lowest energy:', E_real)

one = [[0],[1]]
zero = [[1],[0]]
init = [1]
for i in range(parameters['qubits']):
    if i == parameters['qubits'] // 2:
        init = np.kron(one, init)
    else:
        init = np.kron(zero, init)

sv = np.copy(init)
final_T = 375
M = 1000
E_ests = []
plt.figure(1)
for m in range(1,M + 1):
    t_list = []
    signal = []
    dt = final_T/m
    
    qc_trot_unitary = get_hubb(dt, parameters['qubits'], parameters['T'], parameters['V'])
    time_evol = Operator(qc_trot_unitary).data
    i = 1
    while i*dt <= final_T:
        sv = time_evol @ sv
        # E_test = rayleigh(H, init, sv)
        overlap = (init.conj().T @ sv)[0][0]
        
        t_list.append(dt*i)
        # diff = abs(E_test-E_real)
        # signal.append(diff)
        signal.append(overlap)
        # print(E_test)
        i += 1

    # Windowing
    window = np.hanning(len(signal))
    signal *= window

    # Zero padding
    nfft = 8*len(signal)

    fft_vals = np.fft.fft(signal, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=dt)

    omega = 2 * np.pi * freqs

    fft_vals = np.fft.fftshift(fft_vals)
    omega = np.fft.fftshift(omega)

    energy_axis = -omega
    spectrum = np.abs(fft_vals)

    idx = np.argsort(energy_axis)
    energy_axis = energy_axis[idx]
    spectrum = spectrum[idx]

    mask = (energy_axis >= np.min(H_eigs) - 0.2) & (energy_axis <= np.max(H_eigs) + 0.2)
    spectrum_local = spectrum[mask]/max(spectrum[mask])
    energy_local = energy_axis[mask]
    plt.plot(energy_local, abs(spectrum_local), label=fr"$T_{{\max}}={final_T}$")
    max_ids = sp.signal.find_peaks(spectrum_local, height = 0.5)[0]
    if len(max_ids) == 0:
        E_ests.append(np.nan)
    else:
        E_est_idx = max_ids[0]
        E_ests.append(energy_local[E_est_idx])

    # plt.plot(energy_axis[mask][E_est_idx], spectrum_local[E_est_idx], 'rx', label='peak point')
    # E_est = energy_axis[mask][E_est_idx]

# Exact energies
E_ests = np.array(E_ests)
for j, e in enumerate(H_eigs):
    plt.axvline(e, linestyle="--", alpha=0.5,
                label="exact energies" if j == 0 else None)
plt.xlabel("Energy")
plt.ylabel("Spectral weight")
# plt.legend()
plt.tight_layout()
plt.savefig('MWE_plots/freq_spec.pdf')

plt.figure(2)
plt.scatter(range(1, M+1), E_ests, label = 'Energy')
# plt.yscale('log')
plt.title('Energy Est vs time steps')
plt.xlabel('M')
plt.ylabel('Energy')
plt.legend()
plt.savefig('MWE_plots/Energy_vs_t.pdf')

E_ests = (abs(E_ests-E_real))

plt.figure(3)
plt.plot(range(1, M+1), E_ests, label = 'Energy Diff')
plt.plot(range(1, M+1), [1E-3] * len(range(1, M+1)), label='chemical accuracy')
plt.yscale('log')
plt.title('Zoomed Energy Difference vs t X gate init')
plt.xlabel('M')
plt.ylabel('Energy Difference')
plt.legend()
plt.savefig('MWE_plots/Energy_Diff_vs_t.pdf')