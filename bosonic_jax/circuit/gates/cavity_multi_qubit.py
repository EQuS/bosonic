"""
Gates specific to cavity coupled to multiple ancilla system.
"""

from typing import List, Optional

from bosonic_jax.circuit.base import BosonicGate
import jaxquantum as jqt

import jax.numpy as jnp
import jax.scipy as jsp


class DistributedCDGate(BosonicGate):
    label = "DCD"

    def get_H(self) -> List:

        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        for _, indx in enumerate(self.bqubit_indxs[1:-1]):
            N2 = self.bcirc.breg[indx].params["N"]
            assert N2 == 2, ValueError(
                "Please use a two level system for your control qubits."
            )
        num_ancilla = len(self.bqubit_indxs) - 1
        beta_i = self.params["beta"] / num_ancilla
        chi_i = beta_i / self.ts[-1]

        a = jqt.destroy(N)
        Hs_base = [1.0j * (chi_i * jqt.dag(a) - jnp.conj(chi_i) * a) / jnp.sqrt(2)] + [
            jqt.identity(2) for _ in range(num_ancilla)
        ]
        H_tot = None
        for i in range(1, num_ancilla + 1):
            Hs = Hs_base.copy()
            Hs[i] = jqt.sigmaz() / 2  # D(beta_i)|g><g| + D(0)|e><e|
            H_tot_i = self.extend_gate(Hs)
            H_tot = H_tot_i if H_tot is None else H_tot + H_tot_i

        return [H_tot]

    def get_U(self) -> jnp.ndarray:
        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        for _, indx in enumerate(self.bqubit_indxs[1:-1]):
            N2 = self.bcirc.breg[indx].params["N"]
            assert N2 == 2, ValueError(
                "Please use a two level system for your control qubits."
            )
        num_ancilla = len(self.bqubit_indxs) - 1

        beta_i = self.params["beta"] / num_ancilla

        a = jqt.destroy(N)
        Hs_base = [
            -1.0j * (beta_i * jqt.dag(a) - jnp.conj(beta_i) * a) / jnp.sqrt(2)
        ] + [jqt.identity(2) for i in range(num_ancilla)]
        H_tot = None
        for i in range(1, num_ancilla + 1):
            Hs = Hs_base.copy()
            Hs[i] = jqt.sigmaz() / 2  # D(beta_i)|g><g| + D(-beta_i)|e><e|
            H_tot_i = self.extend_gate(Hs)
            H_tot = H_tot_i if H_tot is None else H_tot + H_tot_i
        U_tot = jsp.linalg.expm(1.0j * H_tot)
        return U_tot


class MultiQubitRxGate(BosonicGate):
    """
    Rₓ(θ) = cos(θ/2)I - isin(θ/2)X
    """

    label = "MRₓ(θ)"

    def get_H(self) -> Optional[List]:
        return None

    def get_U(self) -> jnp.ndarray:
        for _, indx in enumerate(self.bqubit_indxs[0:-1]):
            N2 = self.bcirc.breg[indx].params["N"]
            assert N2 == 2, ValueError("Please use a two level system for your qubits.")

        num_qubits = len(self.bqubit_indxs)
        theta = self.params["theta"]
        I = self.extend_gate([jqt.identity(2) for _ in range(num_qubits)])
        X = self.extend_gate([jqt.sigmax() for _ in range(num_qubits)])
        return jnp.cos(theta / 2) * I - 1.0j * jnp.sin(theta / 2) * X
