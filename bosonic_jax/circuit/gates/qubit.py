""" Fock Qubit gates. """

from typing import List, Optional

import jax.numpy as jnp

from bosonic_jax.circuit.base import BosonicGate
import jaxquantum as jqt


class HGate(BosonicGate):
    """HGate."""

    label = "H"

    def get_H(self) -> Optional[List]:
        # TODO: implement this
        return [0]

    def get_H_func(self, t: float) -> jqt.Qarray:
        raise NotImplementedError("No Hamiltonian for HGate.")

    def get_U(self) -> jqt.Qarray:
        Us = [jqt.hadamard()]
        U_tot = self.extend_gate(Us)
        return U_tot
