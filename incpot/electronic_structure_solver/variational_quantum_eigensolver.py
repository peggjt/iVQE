from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver


class VariationalQuantumEigensolver:

    def __init__(self):
        pass

    def main(self, molecule, remove_orbitals=None):
        # Form molecule
        molecule = PySCFDriver(
            atom=molecule.atom,
            basis=molecule.basis,
            charge=molecule.charge,
            spin=molecule.spin,
            unit=DistanceUnit.ANGSTROM,
        )
        molecule = molecule.run()

        # Apply the FreezeCoreTransformer
        # When freeze_core is enabled (the default), the “core” orbitals will be 
        # determined automatically according to count_core_orbitals.Additionally, 
        # unoccupied spatial orbitals can be removed via a list of indices passed to 
        # remove_orbitals. It is the user’s responsibility to ensure that these are 
        # indeed unoccupied orbitals, as no checks are performed.
        transformer = FreezeCoreTransformer(remove_orbitals=remove_orbitals)
        molecule = transformer.transform(molecule)
        print(
            "Number of spin orbitals after freezing core:", molecule.num_spin_orbitals
        )
        print("Number of particles after freezing core:", molecule.num_particles)

        # Ansatz
        mapper = JordanWignerMapper()
        ansatz = UCCSD(
            molecule.num_spatial_orbitals,
            molecule.num_particles,
            mapper,
            initial_state=HartreeFock(
                molecule.num_spatial_orbitals,
                molecule.num_particles,
                mapper,
            ),
        )

        # Variational Quantum Eigensolver
        vqe_solver = VQE(Estimator(), ansatz, SLSQP())
        vqe_solver.initial_point = [0.0] * ansatz.num_parameters

        # Ground-State Eigensolver
        calc = GroundStateEigensolver(mapper, vqe_solver)
        res = calc.solve(molecule)
        results = {"energy_total": res.total_energies}

        return results
