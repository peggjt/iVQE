import unittest
from incpot.molecule import Molecule


class TestMolecule(unittest.TestCase):
    r"""Test Molecule."""

    def test_basic(self):

        molecule = Molecule()
        molecule.coordinates = 'O 0 0 0; H 0 1 0; H 0 0 1'
        molecule.basis = 'ccpvdz'
        molecule.build()        

if __name__ == '__main__':
    unittest.main()
