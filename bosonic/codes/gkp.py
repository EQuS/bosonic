"""
Cat Code Qubit
"""

from typing import Tuple

from bosonic.codes.base import BosonicQubit
import jaxquantum as jqt

import jax.numpy as jnp
import jax.scipy as jsp


class GKPQubit(BosonicQubit):
    """
    GKP Qudit Class.
    """
    name = "gkp"

    def _params_validation(self):
        super()._params_validation()

        if "delta" not in self.params:
            self.params["delta"] = 0.25

        if 'd' not in self.params:
            self.params['d'] = 2

        self.params["l"] = jnp.sqrt(jnp.pi * self.params['d'])
        s_delta = jnp.sinh(self.params["delta"] ** 2)
        self.params["epsilon"] = s_delta * self.params["l"]

    def _gen_common_gates(self) -> None:
        """
        Overriding this method to add additional common gates.
        """
        super()._gen_common_gates()

        # phase space
        self.common_gates["x"] = (self.common_gates["a_dag"] + self.common_gates["a"]) / jnp.sqrt(self.params['d'])
        self.common_gates["p"] = (1.0j * (self.common_gates["a_dag"] - self.common_gates["a"]) / jnp.sqrt(self.params['d']))

        # finite energy
        self.common_gates["E"] = jqt.expm(-self.params["delta"] ** 2* self.common_gates["a_dag"]@ self.common_gates["a"])
        self.common_gates["E_inv"] = jqt.expm(self.params["delta"] ** 2* self.common_gates["a_dag"]@ self.common_gates["a"])

        # axis
        x_axis, z_axis = self._get_axis()
        y_axis = x_axis + z_axis

        # gates
        X_0 = jqt.expm(1.0j * self.params["l"] * jnp.sqrt(2) / self.params['d'] * z_axis)
        Z_0 = jqt.expm(1.0j * self.params["l"] * jnp.sqrt(2) / self.params['d']* x_axis)
        Y_0 = 1.0j * X_0 @ Z_0

        self.common_gates["X"] = self._make_op_finite_energy(X_0)
        self.common_gates["Z"] = self._make_op_finite_energy(Z_0)
        self.common_gates["Y"] = self._make_op_finite_energy(Y_0)

        # symmetric stabilizers and gates
        self.common_gates["Z_s_0"] = self._symmetrized_expm(1.0j * self.params["l"] / self.params['d'] * x_axis)
        self.common_gates["S_x_0"] = self._symmetrized_expm(1.0j * self.params["l"] * z_axis * jnp.sqrt(2))
        self.common_gates["S_z_0"] = self._symmetrized_expm(1.0j * self.params["l"] * x_axis * jnp.sqrt(2))
        self.common_gates["S_y_0"] = self._symmetrized_expm(1.0j * self.params["l"] * y_axis * jnp.sqrt(2))

    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Construct basis states |+z> and |-z> for the GKP qubit.

        Returns:
            Tuple[jqt.Qarray, jqt.Qarray]: The |+z> and |-z> basis states.
        """
        H_0 = (
            -self.common_gates["S_x_0"]
            - self.common_gates["S_y_0"]
            - self.common_gates["S_z_0"]
            - self.common_gates["Z_s_0"]  # bosonic |+z> state
        )

        _, vecs = jnp.linalg.eigh(H_0.data)
        gstate_ideal = jqt.Qarray.create(vecs[:, 0])

        # step 2: make ideal eigenvector finite energy
        gstate = self.common_gates["E"] @ gstate_ideal

        plus_z = jqt.unit(gstate)
        minus_z = self.common_gates["X"] @ plus_z
        return plus_z, minus_z

    # utils
    # ======================================================
    def _get_axis(self):
        x_axis = self.common_gates["x"]
        z_axis = -self.common_gates["p"]
        return x_axis, z_axis

    def _make_op_finite_energy(self, op):
        return self.common_gates["E"] @ op @ self.common_gates["E_inv"]

    def _symmetrized_expm(self, op):
        return (jqt.expm(op) + jqt.expm(-1.0 * op)) / 2.0

    # gates
    # ======================================================
    @property
    def x_U(self) -> jqt.Qarray:
        return self.common_gates["X"]

    @property
    def y_U(self) -> jqt.Qarray:
        return self.common_gates["Y"]

    @property
    def z_U(self) -> jqt.Qarray:
        return self.common_gates["Z"]


class RectangularGKPQubit(GKPQubit):
    def _params_validation(self):
        super()._params_validation()
        if "a" not in self.params:
            self.params["a"] = 0.8

    def _get_axis(self):
        a = self.params["a"]
        x_axis = a * self.common_gates["x"]
        z_axis = -1 / a * self.common_gates["p"]
        return x_axis, z_axis


class SquareGKPQubit(GKPQubit):
    def _params_validation(self):
        super()._params_validation()
        self.params["a"] = 1.0


class HexagonalGKPQubit(GKPQubit):
    def _get_axis(self):
        a = jnp.sqrt(2 / jnp.sqrt(3))
        x_axis = a * (
            jnp.sin(jnp.pi / 3.0) * self.common_gates["x"]
            + jnp.cos(jnp.pi / 3.0) * self.common_gates["p"]
        )
        z_axis = a * (-self.common_gates["p"])
        return x_axis, z_axis


## Citations

# Stabilization of Finite-Energy Gottesman-Kitaev-Preskill States
# Baptiste Royer, Shraddha Singh, and S. M. Girvin
# Phys. Rev. Lett. 125, 260509 – Published 31 December 2020

# Quantum error correction of a qubit encoded in grid states of an oscillator.
# Campagne-Ibarcq, P., Eickbusch, A., Touzard, S. et al.
# Nature 584, 368–372 (2020).
