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
    h_circ.rxx(2*J_x*dt, 0, 1)
    h_circ.ryy(2*J_x*dt, 0, 1)
    
    # Interaction (ZZ + Z + Z)
    i_circ = QuantumCircuit(2)
    i_circ.rz(-2*B*dt, 0)
    i_circ.rz(-2*B*dt, 1)
    i_circ.rzz(2*B*dt, 0, 1)
    
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

def number_operator(n_qubits):
    N = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for i in range(n_qubits):
        Z = 1
        for j in range(n_qubits):
            if i == j:
                Z = np.kron(Z, np.array([[1,0],[0,-1]]))
            else:
                Z = np.kron(Z, np.eye(2))
        N += 0.5 * (np.eye(2**n_qubits) - Z)
    return N

def quadratic_peak(x, y, i):
    """Sub-bin peak location from 3-point quadratic interpolation."""
    if i <= 0 or i >= len(y) - 1:
        return x[i]

    y1, y2, y3 = y[i - 1], y[i], y[i + 1]
    x1, x2, x3 = x[i - 1], x[i], x[i + 1]

    denom = y1 - 2.0 * y2 + y3
    if abs(denom) < 1e-14:
        return x2

    dx = x2 - x1
    delta = 0.5 * (y1 - y3) / denom
    delta = np.clip(delta, -1.0, 1.0)

    return x2 + delta * dx

def make_trotter_plot(sites_list:list[int]):
    final_times = [2.69, 28.04, 28.04, 22.97, 39.87] # sites 1-5 (qubits 2-10)
    M = 100
    E_approx_list = []
    error = []
    for j, sites in enumerate(sites_list): 
        parameters = {}
        parameters['sys']      = "HUBB"
        parameters['qubits']   = 2*sites
        #HUBB
        parameters['T']        = 1
        parameters['V']        = 1.5

        n_target = parameters['qubits']/2 # half filling
        T = final_times[j]

        H = create_hamiltonian(parameters, scale=True)
        E, V = np.linalg.eigh(H)

        N_op = number_operator(parameters['qubits'])
        # H -= 1e-9*N_op

        # Analyze spectrum 
        E_targets = []
        print("Energy eigenvalues:")
        for k, e in enumerate(E):
            n_expect = np.real(V[:,k].conj().T @ (N_op @ V[:,k]))
            # print(f'eigenvalue {k}, energy {e:.3f}, particle number {np.round(n_expect)}')
            if np.abs(n_expect - n_target) < 1e-3:
                E_targets.append(e)

        E_targets = np.array(E_targets)
        
        print(f'Energies of {n_target}-particle states:')
        print(E_targets)
        n_min = np.argmin(E_targets)

        print(f'Lowest energy {E_targets[n_min]}')

        one = [[0],[1]]
        zero = [[1],[0]]

        init = [1]
        for i in range(parameters['qubits']):
            if -n_target/2 <= i-parameters['qubits']//2 and i-parameters['qubits']//2 < n_target/2:
                print('1')
                init = np.kron(one, init)
            else:
                print('0')
                init = np.kron(zero, init)
        
        psi1_init = init/np.linalg.norm(init)
        overlaps = (np.abs(V.conj().T @ psi1_init) ** 2)
        print("Projected-state spectral weights:")
        # for n, w in enumerate(overlaps):
        #     print(f"n={n}: {w:.6f}")

        target_E = E_targets[n_min]

        E_approxs = []
        errors = []
        for m in range(1, M + 1):
            # nsteps = int(np.ceil(T / 0.1)+1)
            times = np.linspace(0.0, T, m)
            dt = T/m
            # print(T, dt)

            # U_dt = expm(-1j * H * dt)
            qc_trot_unitary = get_hubb(dt, parameters['qubits'], parameters['T'], parameters['V'])
            U_dt = Operator(qc_trot_unitary).data

            psi = psi1_init.copy()
            overlap = []

            for _ in range(m):
                # autocorrelation in the projected subspace
                overlap.append(np.vdot(psi1_init, psi))
                psi = U_dt @ psi

            overlap = np.array(overlap)

            # Window
            window = np.hanning(len(overlap))
            signal = overlap * window

            # Zero padding
            nfft = 8 * len(signal)

            fft_vals = np.fft.fft(signal, n=nfft)
            freqs = np.fft.fftfreq(nfft, d=dt)
            omega = 2.0 * np.pi * freqs

            fft_vals = np.fft.fftshift(fft_vals)
            omega = np.fft.fftshift(omega)

            energy_axis = -omega
            spectrum = np.abs(fft_vals)

            idx = np.argsort(energy_axis)
            energy_axis = energy_axis[idx]
            spectrum = spectrum[idx]

            # Since ground state is removed, E1 is now the lowest strong peak.
            # Use a broad window around E1 and refine with quadratic interpolation.
            window_half_width = max(0.05, 3.0 * 2.0 * np.pi / T)

            mask = (
                (energy_axis >= target_E - window_half_width)
                & (energy_axis <= target_E + window_half_width)
            )

            energy_local = energy_axis[mask]
            spectrum_local = spectrum[mask]

            if len(energy_local) < 3:
                E_approx = np.nan
            else:
                i_max = np.argmax(spectrum_local)
                E_approx = quadratic_peak(energy_local, spectrum_local, i_max)

            E_approxs.append(E_approx)
            errors.append(abs(E_approx - target_E))
        E_approx_list.append(E_approxs)
        error.append(errors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im = axes[0].imshow(error, aspect='auto')

    cbar = axes[0].figure.colorbar(im, ax=axes[0])
    cbar.ax.set_ylabel("Error", rotation=-90, va="bottom")

    axes[0].set_xticks(range(M))
    axes[0].set_xticklabels(range(1, M + 1), rotation=45, ha="right", rotation_mode="anchor")
    axes[0].set_yticks(range(len(sites_list)))
    axes[0].set_yticklabels(sites_list)

    # for i in range(1, sites + 1):
    #     for j in range(1, sites + 1):
    #         text = ax.text(j, i, error[i, j],
    #                     ha="center", va="center", color="w")

    axes[0].set_xlabel("Trotter Steps (M)")
    axes[0].set_ylabel("Number of Sites (L)")
    axes[0].set_title(f"Hubbard ({parameters['qubits'] // 2} sites, T={parameters['T']}, V={parameters['V']} | targeted particle number {n_target})")

    axes[1].plot(range(1, M+1), error[-1], label = 'Energy Diff')
    axes[1].plot(range(1, M+1), [1E-3] * len(range(1, M+1)), label='chemical accuracy')
    axes[1].set_yscale('log')
    axes[1].set_title('Energy Difference vs t')
    axes[1].set_xlabel('M')
    axes[1].set_ylabel('Energy Difference')
    axes[1].legend(loc=3)
    fig.tight_layout()
    plt.savefig('testing_graphs/Hubb_conv.pdf')


if __name__ == "__main__":
    make_trotter_plot(range(1,4))