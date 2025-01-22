import unittest
from ivqe.molecule import Molecule
from ivqe.electronic_structure_solver.variational_quantum_eigensolver import VariationalQuantumEigensolver


class TestMolecule(unittest.TestCase):
    r"""Test Molecule."""

    def test_basic(self):

        molecule = Molecule()
        molecule.coordinates = 'O 0 0 0; H 0 1 0; H 0 0 1'
        molecule.basis = 'sto3g'
        molecule = molecule.build()

        vqe = VariationalQuantumEigensolver(molecule=molecule)

        results = []
        active_occupied_orbitals = [1,2,3,4]
        for i in range(len(active_occupied_orbitals)):
            remove_orbitals = [orbital for j, orbital in enumerate(active_occupied_orbitals) if j != i]
            print(remove_orbitals)
            result = vqe.frozen=remove_orbitals
            result = vqe.run()
            print(result["energy_total"])
            results.append(result["energy_total"])

        for i in results:
            print(i)

        vqe.frozen=[2,3]
        result = vqe.run()
        print(result["energy_total"])

if __name__ == '__main__':
    unittest.main()
