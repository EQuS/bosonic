"""
Gates specific to cavity qubit system.
"""

from typing import List

from bosonic_jax.codes.bcirc.base import BosonicGate
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


class KerrEchoCDGate(BosonicGate):
    """
    Modified experiment-specific Conditional Displacement Gate CD(β) with additional
    Hamiltonian terms due to unwanted residual cavity Kerr.
    """

    label = "CD-Echo"

    def get_H(self) -> List:

        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        N2 = self.bcirc.breg[self.bqubit_indxs[1]].params["N"]
        assert N2 == 2, ValueError(
            "Please use a two level system for your second qubit."
        )
        beta = self.params["beta"]
        r = self.params["r"]
        c = self.params["c"]

        a = jqt.destroy(N)
        a_dag = jqt.dag(a)

        # fmt: off
        H_CD = 1.0j * (beta * a_dag - jnp.conj(beta) * a) / jnp.sqrt(2)
        H_Kerr = (
            1.0j * r * (beta * (a_dag @ a_dag @ a - c * a_dag)
            - jnp.conj(beta) * (a_dag @ a @ a - c * a)) / jnp.sqrt(2)
        )
        H_tot = self.extend_gate([H_CD + H_Kerr, jqt.sigmaz() / 2])
        # fmt: on

        return [H_tot]

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


class NumSplitCDGate(BosonicGate):
    """
    Modified experiment-specific Conditional Displacement Gate CD(β) with number-split
    driving on the ancilla qubit g and e peaks. Makes using of the dispersive coupling
    between a cavity and 2-level qubit: χ(a'a)σz. This results in a characteristic gate
    time τ_CD = π / χ. In this case, we need epsilon = -1j * β / jnp.sqrt(2) to match
    the unitary case.

    #TODO: Generalize relation between epsilon, beta, and τ_CD.
    """

    label = "CD-Num-Split"

    def get_H_func(self, t: float) -> jnp.ndarray:
        """
        H(t)

        Args:
            t (float): time
        
        Returns:
            jnp.ndarray
        """
        Hs = self.get_H()
        H1 = Hs[1][0]
        H2 = Hs[2][0]
        H3 = Hs[3][0]
        H4 = Hs[4][0]

        chi = self.params["chi"]

        return (H1 + H4) * jnp.exp(-1j * chi * t) + (H2 + H3) * jnp.exp(1j * chi * t)

    def get_H(self) -> List:

        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        N2 = self.bcirc.breg[self.bqubit_indxs[1]].params["N"]
        assert N2 == 2, ValueError(
            "Please use a two level system for your second qubit."
        )

        a = jqt.destroy(N)
        a_dag = jqt.dag(a)
        I_q = jqt.identity(N2)

        # Relate ε to β, assuming τ_CD = π/χ
        epsilon = 1j * self.params["beta"] / jnp.sqrt(2)

        epsilon_1 = epsilon
        epsilon_2 = -1 * epsilon
        chi = self.params["chi"]

        args = {}
        args["chi"] = chi
        self.args = args

        H1 = self.extend_gate([epsilon_1 * a_dag, I_q])
        H2 = self.extend_gate([jnp.conj(epsilon_1) * a, I_q])
        H3 = self.extend_gate([epsilon_2 * a_dag, I_q])
        H4 = self.extend_gate([jnp.conj(epsilon_2) * a, I_q])

        return [
            0,
            [H1, "exp(-1j * chi * t)"],
            [H2, "exp(1j * chi * t)"],
            [H3, "exp(1j * chi * t)"],
            [H4, "exp(-1j * chi * t)"],
        ]

    def get_U(self) -> jnp.ndarray:
        """
        This is the same unitary as in CDGate!
        """
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

            if a.size != 3 or norm == 0.0:
                raise ValueError(
                    "Please use a nonzero Bloch vector of the form a = [ax, ay, az] ≠ [0,0,0]."
                )
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


class NumDepQubitRotationGate(QubitRotationGate):
    label = "Photon-Number Dependent Qubit Rotation"

    def get_H(self) -> List:
        N = self.bcirc.breg[self.bqubit_indxs[0]].params["N"]
        omega = self.params["omega"]
        eps_rot = self.params["eps_rot"]
        sigma_tot = self.get_sigma_tot()

        I_a = jqt.identity(N)
        I_q = jqt.identity(2)

        H_tot = self.extend_gate([I_a, 0 * I_q])
        for n in range(N):
            H_tot += self.extend_gate(
                [jqt.ket2dm(jqt.basis(N, n)), sigma_tot * (omega + eps_rot[n]) / 2]
            )

        return [H_tot]
