# Copyright (c) James Pegg.
# Distributed under the terms of the MIT License.

import unittest

from incpot.molecule import Molecule
from incpot.incremental_decomposition import IncrementalDecomposition


class TestIncrementalDecomposition(unittest.TestCase):
    r"""Test Self-Consistent Fields."""

    def test_basic_ccsd(self):
        r"""Test restricted Hartree-Fock mean-field."""
        molecule = Molecule()
        molecule.coordinates = "O 0 0 0; H 0 1 0; H 0 0 1"
        molecule.basis = "ccpvdz"
        molecule = molecule.build()

        idm = IncrementalDecomposition()
        idm.post_hf_method = "ccsd"
        idm.n_body_truncation = 3
        idm.execute(molecule)

    def test_basic_cisd(self):
        r"""Test restricted Hartree-Fock mean-field."""
        molecule = Molecule()
        molecule.coordinates = "O 0 0 0; H 0 1 0; H 0 0 1"
        molecule.basis = "ccpvdz"
        molecule = molecule.build()

        idm = IncrementalDecomposition()
        idm.post_hf_method = "cisd"
        idm.n_body_truncation = 3
        results = idm.execute(molecule)

        print(results)

    def test_basic_variational_quantum_eigensolver(self):
        r"""Test restricted Hartree-Fock mean-field."""
        molecule = Molecule()
        molecule.coordinates = "O 0 0 0; H 0 1 0; H 0 0 1"
        molecule.basis = "sto3g"
        molecule = molecule.build()

        idm = IncrementalDecomposition()
        idm.post_hf_method = "vqe"
        idm.n_body_truncation = 2
        results = idm.execute(molecule)

        print(results)

    def test_fragment_screening(self):
        r"""Test fragment screening"""
        molecule = Molecule()
        molecule.coordinates = """
            C	 0.0000	 1.3970	 0.0000
            C	 1.2098	 0.6985	 0.0000
            C	 1.2098	-0.6985	 0.0000
            C	 0.0000	-1.3970	 0.0000
            C	-1.2098	-0.6985	 0.0000
            C	-1.2098	 0.6985	 0.0000
            H	 0.0000	 2.4810	 0.0000
            H	 2.1486	 1.2405	 0.0000
            H	 2.1486	-1.2405	 0.0000
            H	 0.0000	-2.4810	 0.0000
            H	-2.1486 -1.2405	 0.0000
            H	-2.1486	 1.2405	 0.0000
        """
        molecule.basis = "3-21G"
        molecule = molecule.build()

        idm = IncrementalDecomposition()
        idm.post_hf_method = "ccsd"
        idm.n_body_truncation = 3
        idm.fragment_threshold = 0.004
        results = idm.execute(molecule)

        print(results)

if __name__ == "__main__":
    unittest.main()
