from typing import Optional, Sequence, Iterable
import numpy as np
import numpy.typing as npt
import math
from pyscf import gto, scf, cc
from pyscf.gto import Mole
from quri_parts.pyscf.mol import get_spin_mo_integrals_from_mole
from quri_parts.openfermion.mol import get_qubit_mapped_hamiltonian
from quri_parts.openfermion.ansatz import TrotterUCCSD
from quri_parts.core.state import quantum_state, apply_circuit
from quri_vm import VM
from quri_parts.backend.devices import star_device
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts
from quri_parts_qsci import qsci
from quri_parts.chem.mol import ActiveSpace, cas


class QSCI:

    def __init__(
        self,
        trotter_steps: int = 1,
        use_singles: bool = True,
        reduce_parameter: bool = True,
        total_shots: int = 10,
        logical_error_rate: float = 1e-6,
        physical_error_rate: float = 1e-4,
        threshold_error_rate: float = 1e-2,
        num_states_pick_out: int = 50
    ):
        r"""

        Args:
            trotter_steps: The number of steps the UCCSD unitary operator is approximated by Trotter decomposition
            use_singles: Whether single excitations (T₁) are included in the UCCSD ansatz.
            reduce_parameter: Reduce the number of variational parameters in UCCSD. This option exploits spin and
                spatial symmetries (e.g., restricting excitations to singlet pairs only), significantly reducing the
                number of parameters.
            total_shots: The total number of measurement shots used in the QSCI procedure.
            logical_error_rate (float): Desired logical error rate after error correction.
                Defaults to 1e-6 (one failure per million operations).
            physical_error_rate (float): Physical error rate per gate or operation.
                Must be less than the threshold_error_rate. Defaults to 1e-4 (0.01%).
            threshold_error_rate (float): Error threshold of the quantum error-correcting code.
                For typical surface codes, around 1%. Defaults to 1e-2.
            num_states_pick_out: number of states to pick out.
        """
        self.trotter_steps = trotter_steps
        self.use_singles = use_singles
        self.reduce_parameter = reduce_parameter
        self.total_shots = total_shots
        self.logical_error_rate = logical_error_rate
        self.physical_error_rate = physical_error_rate
        self.threshold_error_rate = threshold_error_rate
        self.num_states_pick_out=num_states_pick_out

    def run(self, molecule: Mole, frozen: Optional[Sequence[int]] = None) -> None:
        """
        Run the QSCI (Quantum Selected Configuration Interaction) workflow for a given molecule.

        This method performs the following steps:
          1. Sets up the mean-field (HF) calculation with optional frozen orbitals.
          2. Constructs the qubit-mapped Hamiltonian in an active space.
          3. Builds a UCCSD quantum circuit using classical CCSD amplitudes.
          4. Prepares a quantum state with these parameters.
          5. Simulates sampling with a virtual quantum device.
          6. Applies the QSCI method to estimate ground-state energy.

        Args:
            molecule (Mole): A PySCF Mole object defining the molecular system
                (geometry, basis set, etc.).
            frozen (Optional[Sequence[int]]): List of orbital indices to freeze
                (i.e., exclude from the correlated active space). If None, no orbitals are frozen.
        """
        # Define Orbital Indices.
        n_frozen = len(frozen) if frozen else 0
        print("n_frozen", n_frozen)
        n_orbitals = molecule.nao * 2 - 2 * n_frozen
        n_electrons = molecule.nelectron - 2 * n_frozen
        active_orbs_indices = None
        if frozen is not None:
            active_orbs_indices = [i for i in range(molecule.nao) if i not in frozen]

        # Find Hamiltonian.
        mf = scf.RHF(molecule).run(verbose=0)
        active_space = cas(
            n_active_ele=n_electrons,
            n_active_orb=n_orbitals // 2,
            active_orbs_indices=active_orbs_indices,
        )
        print(active_space)
        print(active_orbs_indices)

        hamiltonian, mapping = get_qubit_mapped_hamiltonian(
            *get_spin_mo_integrals_from_mole(molecule, mf.mo_coeff, active_space)
        )

        # Find bound-state.
        uccsd = TrotterUCCSD(
            n_orbitals,
            n_electrons,
            trotter_number=self.trotter_steps,
            use_singles=self.use_singles,
            singlet_excitation=self.reduce_parameter,
        )
        ccsd = cc.CCSD(mf)
        ccsd.frozen = frozen
        ccsd.kernel()
        param = self.ccsd_param_to_circuit_param(uccsd, n_electrons, ccsd.t1, ccsd.t2)

        hf_state = quantum_state(n_orbitals, bits=2**n_electrons - 1)
        state = apply_circuit(uccsd, hf_state)
        bound_state = state.bind_parameters(param)

        # Define Sampler.
        star_vm = VM.from_device_prop(
            star_device.generate_device_property(
                qubit_count=2*len(active_orbs_indices),
                code_distance=self.code_distance(),
                qec_cycle=TimeValue(1, TimeUnit.MICROSECOND),
                physical_error_rate=self.physical_error_rate,
            )
        )
        sampler = self.create_concurrent_sampler_from_vm(star_vm)

        # Run QSCI.
        eigs, _ = qsci(
            hamiltonian,
            [bound_state],
            sampler,
            total_shots=self.total_shots,
            num_states_pick_out=self.num_states_pick_out,
        )

        print(f"The qsci gs energy is: {eigs[0]}")

    def code_distance(self):
        r"""
        Estimate the minimum surface code distance required to suppress physical errors
        down to a target logical error rate.

        This function uses the approximate scaling law of the logical error rate in surface
        codes:
            ε_L ≈ A * (ε_P / ε_th)^{(d+1)/2}
        where ε_L is the logical error rate, ε_P the physical error rate, ε_th the code threshold,
        and d is the code distance. The constant A is assumed to be near 1.

        Returns:
            int: Recommended minimum code distance (odd integer) needed to achieve the target
                 logical error rate under the given physical error rate.

        """
        if self.physical_error_rate >= self.threshold_error_rate:
            raise ValueError("Physical error rate must be below the threshold.")

        log_ratio = math.log(self.logical_error_rate) / math.log(self.physical_error_rate / self.threshold_error_rate)
        code_distance = max(1, int(math.ceil(2 * log_ratio - 1)))

        # Ensure d is odd (as typical for surface codes).
        if code_distance % 2 == 0:
            code_distance += 1

        return code_distance

    def create_concurrent_sampler_from_vm(
        self,
        vm: VM,
    ) -> ConcurrentSampler:
        """Create a simple :class:`~ConcurrentSampler` using a :class:`~SamplingBackend`.

        Defines a wrapper around a virtual quantum machine (VM) that conforms to the ConcurrentSampler interface
        expected by the quri_parts QSCI and estimation modules.
        """

        def sampler(
            shot_circuit_pairs: Iterable[tuple[ImmutableQuantumCircuit, int]]
        ) -> Iterable[MeasurementCounts]:
            jobs = [
                vm.sample(circuit, n_shots) for circuit, n_shots in shot_circuit_pairs
            ]
            return map(lambda j: j, jobs)

        return sampler

    def ccsd_param_to_circuit_param(
        self,
        uccsd: TrotterUCCSD,
        n_electrons: int,
        t1: npt.NDArray[np.complex128],
        t2: npt.NDArray[np.complex128],
    ) -> Sequence[float]:
        """
        Convert classical CCSD amplitudes (t1, t2) into the parameter list expected by a Trotterized UCCSD quantum
        circuit.

        This method maps the CCSD single (t1) and double (t2) excitation amplitudes from PySCF to the parameter order
        defined by the TrotterUCCSD circuit's `param_mapping.in_params`.

        The input circuit uses spin-orbital indices for singles and doubles, and adjusts the virtual orbital indexing by
        subtracting `n_electrons // 2` to align with the PySCF occupied/virtual partitioning.

        Args:
            uccsd (TrotterUCCSD): The UCCSD ansatz whose parameters are to be set.
            n_electrons (int): The number of electrons in the active space, used
                to determine the occupied and virtual orbital split.
            t1 (np.ndarray): The single excitation amplitudes from PySCF CCSD,
                shape (n_occ, n_vir).
            t2 (np.ndarray): The double excitation amplitudes from PySCF CCSD,
                shape (n_occ, n_occ, n_vir, n_vir).

        Returns:
            Sequence[float]: A list of real-valued parameters matching the expected order of the UCCSD circuit.
        """
        in_param_list = uccsd.param_mapping.in_params
        param_list = []

        for param in in_param_list:
            name_split = param.name.split("_")
            if name_split[0] == "s":
                _, i_str, j_str = name_split
                i, j = int(i_str), int(j_str) - n_electrons // 2
                param_list.append(t1[i, j])

            if name_split[0] == "d":
                _, i_str, j_str, a_str, b_str = name_split
                i, j, b, a = (
                    int(i_str),
                    int(j_str),
                    int(b_str) - n_electrons // 2,
                    int(a_str) - n_electrons // 2,
                )
                param_list.append(t2[i, j, a, b])
        return param_list


molecule = gto.M(atom="O 0 0 0; H 0 0 1; H 1 0 0", basis="sto3g")
qsci_method = QSCI()
qsci_method.run(molecule=molecule, frozen=[0, 4])
