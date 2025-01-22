"""
Holds the Molecule class.

"""

from pyscf import gto


class Molecule():

    def __init__(self,
                 coordinates: str = None,
                 basis: str = None):
        r"""The init function."""
        self.coordinates = coordinates
        self.basis = basis

    def build(self):
        r"""Builds the molecule.

        Returns:
            molecule
        """
        molecule = gto.M()
        molecule.unit = "Angstrom"
        if not self.coordinates:
            raise ValueError(r'No `coordinates` found.')
        if not self.basis:
            raise ValueError(r'No `basis` found.')

        molecule.atom = self.coordinates
        molecule.basis = self.basis
        molecule.build()

        return molecule
