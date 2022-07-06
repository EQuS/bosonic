"""
Common Gates
"""
from typing import List

from bosonic_jax.codes.bcirc.base import BosonicGate
import bosonic_jax.jax_qutip as jqt

import jax.numpy as jnp


class DelayGate(BosonicGate):
    label = "I"

    def get_H(self) -> List:
        Hs: List[jnp.ndarray] = []
        H_tot = self.extend_gate(Hs)
        return [H_tot]

    def get_U(self) -> jnp.ndarray:
        Us: List[jnp.ndarray] = []
        U_tot = self.extend_gate(Us)
        return U_tot


class DisplaceGate(BosonicGate):
    label = "D"

    def get_H(self) -> List:
        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        alpha = self.params["alpha"]
        a = jqt.destroy(N)
        Hs = [-1.0j * (alpha * jqt.dag(a) - jnp.conj(alpha) * a) / jnp.sqrt(2)]
        H_tot = self.extend_gate(Hs)
        return [H_tot]


class PhaseRotationGate(BosonicGate):
    label = "Phase Rotation"

    def get_H(self) -> List:
        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        phi = self.params["phi"]
        N_a = jqt.num(N)
        Hs = [phi * N_a]
        H_tot = self.extend_gate(Hs)
        return [H_tot]
