# Copyright (c) James Pegg.
# Distributed under the terms of the MIT License.


"""
Incremental Decomposition.

"""

from itertools import combinations
from pyscf import scf, cc, ci, fci
from ivqe.electronic_structure_solver import VariationalQuantumEigensolver


class IncrementalDecomposition:
    r"""Incremental Decomposition."""

    def __init__(
        self,
        post_hf_method: str = None,
        n_body_truncation: int = None,
        fragment_threshold: float = None,
    ):
        r"""
        Initialise an instance of :class:`.IncrementalDecomposition`.

        Args:
            post_hf_method (`str`): The post Hartree-Fock method.
            n_body_truncation (`int`): The n-body terms.
            fragment_threshold (`float`): The threshold of eliminating n-body (n > 2) fragments.
        """
        self.post_hf_method = post_hf_method
        self.n_body_truncation = n_body_truncation
        self.fragment_threshold = fragment_threshold or 0.0

    def run(self, molecule):
        r"""Run main code.

        Args:
            molecule: (:obj:`pyscf.gto.mole.Mole`): The pyscf molecule class.

        Returns:
            data: (`dict`): A dictionary of results.

        """

        self.natural_orbitals(molecule)
        mean_field, data = self.mean_field(molecule=molecule)
        data = self.n_body(mean_field=mean_field, data=data)

        print()
        print("E_tot: ", data["energy_total"])
        print("E_cor: ", data["energy_correlation"])
        print("E_rhf: ", data["energy_hf"])
        print()

        return data

    def natural_orbitals(self, molecule):
        r"""Find orbital spaces.

        Args:
            molecule: (:obj:`pyscf.gto.mole.Mole`): The pyscf molecule class.

        """

        if not molecule.nelectron % 2 == 0:
            raise ValueError(f"The ground-state must be a singlet.")

        occupied_orbitals = list(i for i in range(molecule.nelectron // 2))
        frozen_occupied_orbitals = list(i for i in range(self.frozen_core(molecule)))
        active_occupied_orbitals = [
            i for i in occupied_orbitals if i not in frozen_occupied_orbitals
        ]

        # Add attributes.
        self.occupied_orbitals = occupied_orbitals
        self.frozen_occupied_orbitals = frozen_occupied_orbitals
        self.active_occupied_orbitals = active_occupied_orbitals

    def frozen_core(self, molecule):
        r"""Find default frozen orbtials.

        Args:
            molecule: (:obj:`pyscf.gto.mole.Mole`): The pyscf molecule class.

        Returns:
            n_frozen_core: (int): The number of frozen core orbitals
        """
        # fmt: off
        frozen_core = {
            "H":  0, "He": 0,
            "Li": 0, "Be": 0,
            "B":  1, "C":  1, "N":  1, "O":  1, "F":  1, "Ne":  1,
        }
        # fmt: on

        n_frozen_core = 0
        for i in molecule.elements:
            n_frozen_core += frozen_core[i]

        return n_frozen_core

    def mean_field(self, molecule):
        r"""Calculate self-consistent field.

        Args:
            molecule: (:obj:`pyscf.gto.mole.Mole`): The pyscf molecule class.

        """
        rhf = scf.HF(mol=molecule).run()
        data = {}
        data["energy_hf"] = rhf.e_tot

        return rhf, data

    def n_body(self, mean_field, data: dict):
        """n-Body Problems.

        Args:
            mean_field: (class 'pyscf.scf.hf.RHF'): The pyscf mean-field class.
            data: (`dict`): A dictionary of results.

        Returns:
            data: (`dict`): A dictionary of results.

        """
        for n in range(1, self.n_body_truncation + 1):
            print(f"n-Body: {n}")
            data[n] = {}
            for i in combinations(self.active_occupied_orbitals, n):
                print(f"Fragment: {i}")
                data[n][i] = {}

                # Screen fragment.
                if self.include_fragment(n_body=n, fragment=i, data=data) is True:

                    # Execute post Hartree-Fock method.
                    frozen = [f for f in self.occupied_orbitals if f not in i]
                    post_hf_method = self.find_post_hf_method(
                        mean_field=mean_field, frozen=frozen
                    )
                    post_hf_method.run()

                    data[n][i]["energy_total"] = post_hf_method.e_tot
                    data[n][i]["energy_correlation"] = (
                        post_hf_method.e_tot - mean_field.e_tot
                    )

                    # Find epsilon energy.
                    data[n][i]["energy_epsilon"] = self.find_energy_epsilon(
                        data=data, fragment=i
                    )

            # Find n-body energy.
            data[n]["energy_correlation"] = self.find_n_body_correlation(
                n_body=n, data=data
            )

        # Find total energy.
        energy_total, energy_correlation = self.find_energy_total(
            mean_field=mean_field, data=data
        )
        data["energy_total"], data["energy_correlation"] = (
            energy_total,
            energy_correlation,
        )

        return data

    def include_fragment(self, n_body: int, fragment: list, data: dict):
        r"""Include fragment.

        Args:
            n_body (`int`): The fragment n-body term.
            fragment (`list`): The fragment orbital indicies.
            data (`dict`): The results dictionary.

        Returns:
            include_fragment (bool): If the fragment should be included.

        """
        include_fragment = True

        if n_body > 2:
            n_connected = 0
            for i in combinations(fragment, n_body - 1):
                energy_epsilon_abs = abs(data[n_body - 1][i]["energy_epsilon"])
                if energy_epsilon_abs > self.fragment_threshold:
                    n_connected += 1
            if 1 > n_connected:
                include_fragment = False
                print(f"Ignoring fragment: {fragment}")

        return include_fragment

    def find_post_hf_method(self, mean_field, frozen: list):
        r"""Form post Hartree-Fock object.

        Args:
            mean_field: (class 'pyscf.scf.hf.RHF'): The pyscf mean-field class.
            frozen (list): The frozen orbital indicies.

        Returns:
            The post Hartree-Fock method.

        """

        post_hf_method = self.post_hf_method.lower()
        if post_hf_method == "ccsd":
            return cc.CCSD(mf=mean_field, frozen=frozen)

        elif post_hf_method == "cisd":
            return ci.CISD(mf=mean_field, frozen=frozen)

        elif post_hf_method in ["variational_quantum_eigensolver", "vqe"]:
            return VariationalQuantumEigensolver(molecule=mean_field.mol, frozen=frozen)

        else:
            raise ValueError(f"The post-HF method is unknown: {self.post_hf_method}.")

    def find_energy_epsilon(self, fragment: list, data: dict):
        r"""Find epsilon energy.

        Args:
            fragment: (`list`): The fragment orbital indicies.
            data: (`dict`): The results dictionary.

        Returns:
            energy_epsilon: (float): The fragment energy epsilon term.

        The `energy_epsilon` term is defined:

            e_i   = Ec
            e_ij  = Ec - sum(e_i, e_j)
            e_ijk = Ec - sum(e_i, e_j, e_ij, e_jk)

        """
        energy_correlation = data[len(fragment)][fragment]["energy_correlation"]

        energy_epsilon = 0
        for n in range(1, len(fragment)):
            for i in combinations(fragment, n):
                energy_epsilon += data[n][i]["energy_epsilon"]

        energy_epsilon = energy_correlation - energy_epsilon

        return energy_epsilon

    def find_n_body_correlation(self, n_body: int, data: dict):
        r"""Find n-body correlation energy.

        Args:
            n_body (`int`): The n-body term.
            data (`dict`): The results dictionary.

        Returns:
            energy_n_body_correlation (`float`): The n-body correlation energy.
        """
        energy_n_body_correlation = 0
        for i in data[n_body]:
            if "energy_epsilon" in data[n_body][i].keys():
                energy_n_body_correlation += data[n_body][i]["energy_epsilon"]

        return energy_n_body_correlation

    def find_energy_total(self, mean_field, data: dict):
        r"""Find total energy.

        Args:
            mean_field: (class 'pyscf.scf.hf.RHF'): The pyscf mean-field class.
            data (`dict`): The results dictionary.

        Returns:
            energy_total (`float`): The total energy.
            energy_correlation (`float`): The total correlation energy.
        """
        energy_correlation = 0
        for n in range(1, self.n_body_truncation+1):
            energy_correlation += data[n]["energy_correlation"]

        energy_total = mean_field.e_tot + energy_correlation

        return energy_total, energy_correlation
