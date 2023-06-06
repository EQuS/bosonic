"""
Gates specific to cavity qubit system.
"""

from typing import List

from bosonic_jax.circuit.base import BosonicGate
import jaxquantum as jqt

import jax.numpy as jnp
import jax.scipy as jsp


class CDGate(BosonicGate):
    """
    Conditional Displacement Gate CD(β). Needs to be run with a ts array going
    from t=0 to t=1 to get correct β.
    # TODO: Maybe add rate scale factor g, so that t runs from 0 to 1/g
    """

    label = "CD"

    def get_H(self) -> List:

        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        N2 = self.bcirc.breg[self.bqubit_indxs[1]].params["N"]
        assert N2 == 2, ValueError(
            "Please use a two level system for your second qubit."
        )
        beta = self.params["beta"]
        a = jqt.destroy(N)
        H_tot = self.extend_gate(
            [
                1.0j * (beta * jqt.dag(a) - jnp.conj(beta) * a) / jnp.sqrt(2),
                jqt.sigmaz() / 2,
            ]
        )

        return [H_tot]

    def get_H_func(self, t: float) -> jnp.ndarray:
        return self.H[0]

    def get_U(self) -> jnp.ndarray:
        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        N2 = self.bcirc.breg[self.bqubit_indxs[1]].params["N"]
        assert N2 == 2, ValueError(
            "Please use a two level system for your second qubit."
        )
        beta = self.params["beta"]
        a = jqt.destroy(N)
        Hs = [
            -1.0j * (beta * jqt.dag(a) - jnp.conj(beta) * a) / jnp.sqrt(2),
            jqt.sigmaz() / 2,
        ]
        H_tot = self.extend_gate(Hs)
        U_tot = jsp.linalg.expm(1.0j * H_tot)
        return U_tot


class QubitRotationGate(BosonicGate):
    """
    Qubit Rotation Gate R_a(θ) about an axis a = [ax, ay, az] of the Bloch
    spheres. If no axis is provided, defaults to to R_x(θ) about x-axis.
    """

    label = "Qubit Rotation"

    def get_sigma_tot(self) -> jnp.ndarray:
        """
        Helper function that returns (a•σ) = ax*σx + ay*σy + az*σz given
        an input Bloch vector a. If none is provided, default to σx.
        """

        if "rot_axis" in self.params.keys():
            a = jnp.array(self.params["rot_axis"])
            norm = jnp.linalg.norm(a)

            # if a.size != 3 or norm == 0.0:
            #     raise ValueError(
            #         "Please use a nonzero Bloch vector of the form a = [ax, ay, az] ≠ [0,0,0]."
            #     )

            a = (1 / norm) * a
            sigma_tot = a[0] * jqt.sigmax() + a[1] * jqt.sigmay() + a[2] * jqt.sigmaz()

        else:
            sigma_tot = jqt.sigmax()

        return sigma_tot

    def get_H(self) -> List:

        omega = self.params["omega"]
        sigma_tot = self.get_sigma_tot()

        H_tot = self.extend_gate([sigma_tot * omega / 2])

        return [H_tot]

    def get_H_func(self, t: float) -> jnp.ndarray:
        return self.H[0]

    def get_U(self) -> jnp.ndarray:
        theta = self.params["theta"]
        sigma_tot = self.get_sigma_tot()

        H_tot = self.extend_gate([sigma_tot * theta / 2])

        return jsp.linalg.expm(-1.0j * H_tot)
