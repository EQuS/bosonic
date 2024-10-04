"""
Gates specific to cavity qubit system.
"""

from typing import List

from bosonic.circuit.base import BosonicGate
import jaxquantum as jqt

import jax.numpy as jnp
import jax.scipy as jsp

class GKPCHGate(BosonicGate):
    """
    Conditional Displacement Gate CD(β). Needs to be run with a ts array going
    from t=0 to t=1 to get correct β.
    # TODO: Maybe add rate scale factor g, so that t runs from 0 to 1/g
    """

    label = "GKP CH"

    def get_H(self) -> List:

        q0 = self.bcirc.breg[self.bqubit_indxs[0]]
        q1 = self.bcirc.breg[self.bqubit_indxs[1]]

        a = q0.common_gates["a"]
        a_dag = q0.common_gates["a_dag"]

        b = q1.common_gates["a"]
        b_dag = q1.common_gates["a_dag"]

        chi = self.params["chi"]
        
        H_tot = self.extend_gate(
            [
                chi * (a_dag + a),
                (b_dag @ b),
            ]
        )

        return [H_tot]

    def get_H_func(self, t: float) -> jqt.Qarray:
        return self.H[0]

    def get_U(self) -> jqt.Qarray:
        q0 = self.bcirc.breg[self.bqubit_indxs[0]]
        q1 = self.bcirc.breg[self.bqubit_indxs[1]]

        a = q0.common_gates["a"]
        a_dag = q0.common_gates["a_dag"]

        b = q1.common_gates["a"]
        b_dag = q1.common_gates["a_dag"]

        angle = jnp.pi/2
        
        H_tot = self.extend_gate(
            [
                angle * (a_dag + a),
                (b_dag @ b),
            ]
        )

        U_tot = jqt.expm(1.0j * H_tot)
        return U_tot