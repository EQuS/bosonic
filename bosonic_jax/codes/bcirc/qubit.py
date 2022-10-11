""" Fock Qubit gates. """

from typing import List, Optional

import jax.numpy as jnp

from bosonic_jax.codes.bcirc.base import BosonicGate

class HGate(BosonicGate):
    """ HGate."""
    label = "H"

    def get_H(self) -> Optional[List]:
        # TODO: implement this
        return [0]

    def get_H_func(self, t: float) -> jnp.ndarray:
        raise NotImplementedError("No Hamiltonian for HGate.")

    def get_U(self) -> jnp.ndarray:
        Us = [jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2)]
        U_tot = self.extend_gate(Us)
        return U_tot

