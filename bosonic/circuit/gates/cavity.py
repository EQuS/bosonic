"""
Common Gates
"""
from typing import List, Optional

from bosonic.circuit.base import BosonicGate
import jaxquantum as jqt

import jax.numpy as jnp


class DelayGate(BosonicGate):
    label = "I"

    def get_H(self) -> List:
        Hs: List[jqt.Qarray] = []
        H_tot = self.extend_gate(Hs)
        return [H_tot]

    def get_H_func(self, t: float) -> Optional[jqt.Qarray]:
        return self.H[0]

    def get_U(self) -> jqt.Qarray:
        Us: List[jqt.Qarray] = []
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

    def get_H_func(self, t: float) -> Optional[jqt.Qarray]:
        return self.H[0]


class PhaseRotationGate(BosonicGate):
    label = "Phase Rotation"

    def get_H(self) -> List:
        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        phi = self.params["phi"]
        N_a = jqt.num(N)
        Hs = [phi * N_a]
        H_tot = self.extend_gate(Hs)
        return [H_tot]

    def get_H_func(self, t: float) -> Optional[jqt.Qarray]:
        return self.H[0]
