"""
Bosonic Quantum Circuit
"""

from typing import List, Tuple, Union, Type, cast, Dict, Optional, Any, Callable
from abc import abstractmethod, ABCMeta
from numbers import Number

from bosonic_jax.codes.base import BosonicQubit
from jaxquantum.utils.utils import is_1d, device_put_params
import jaxquantum as jqt

from jax import device_put
from jax import config
import jax.numpy as jnp
import jax.scipy as jsp

config.update("jax_enable_x64", True)


class BosonicRegister:
    """
    Register of bosonic qubits.
    """

    def __init__(
        self,
        bqubits: List[BosonicQubit],
    ):
        """
        Bosonic Register Init Method

        Args:
            bqubits:
                List of bosonic qubits to store in BosonicRegister
        """
        self.bqubits: List[BosonicQubit] = bqubits

    def __getitem__(self, key: int):
        """
        Allows us to return the nth element of BosonicRegister as a list.
        """
        return self.bqubits[key]

    def __len__(self):
        return len(self.bqubits)


class BosonicCircuit:
    """
    BosonicCircuit allows users to build quantum circuits out of many bosonic qubits.
    """

    def __init__(self, breg: BosonicRegister):
        self.breg = breg
        self.reset_gates()

    def get_dims(self):
        dims = []
        for bq in self.breg.bqubits:
            dims.append(bq.params["N"])
        dims = jnp.array(dims)
        dims = jnp.array([dims, jnp.ones_like(dims)])
        return dims

    @property
    def dims(self):
        return self.get_dims()

    @property
    def dm_dims(self):
        return jnp.array([self.dims[0], self.dims[0]])

    @property
    def default_initial_state(self):
        return self.gen_default_initial_state()

    def gen_default_initial_state(self) -> jnp.ndarray:
        state = None

        for bq in self.breg.bqubits:
            state = (
                bq.basis["+z"] if state is None else jqt.tensor(state, bq.basis["+z"])
            )

        return cast(jnp.ndarray, state)

    def reset_gates(self) -> None:
        self.gates: List[BosonicGate] = []

    def reset(self) -> None:
        self.reset_gates()

    def add(
        self,
        gate_type: Type["BosonicGate"],
        bqubit_indxs: Union[int, Tuple[int, ...]],
        params: Optional[Dict[str, complex]] = None,
        ts: Optional[jnp.ndarray] = None,
        use_unitary: Optional[bool] = False,
    ):
        if type(bqubit_indxs) == int:
            bqubit_indxs = (bqubit_indxs,)
        bqubit_indxs = cast(Tuple[int, ...], bqubit_indxs)

        num_qubits = len(self.breg)
        for j in bqubit_indxs:
            if j >= num_qubits:
                raise ValueError(
                    f"Please choose qubit indices in the range [0,{num_qubits-1}]"
                )

        self.gates.append(
            gate_type(self, bqubit_indxs, params=params, ts=ts, use_unitary=use_unitary)
        )

    def x(self, bqubit_indx: int) -> None:
        """
        Logical X
        """
        self.gates.append(XGate(self, bqubit_indx))

    def y(self, bqubit_indx: int) -> None:
        """
        Logical Y
        """
        self.gates.append(YGate(self, bqubit_indx))

    def z(self, bqubit_indx: int) -> None:
        """
        Logical Z
        """
        self.gates.append(ZGate(self, bqubit_indx))

    def draw(self):
        NotImplementedError("Not implemented yet!")

    def jqt2qt(self, state: jnp.ndarray):
        return jqt.jqt2qt(state)

    def plot(
        self,
        bqubit_indx: Optional[int] = None,
        state: Optional[Union[jnp.ndarray, jnp.ndarray]] = None,
    ):
        """
        Plot default_initial_state or other state of the bcirc.

        Args:
            bqubit_indx (int): index of logical qubit in bcirc

        """
        state = state if state is not None else self.default_initial_state
        state = self.jqt2qt(state)

        if bqubit_indx is not None:
            self.breg[bqubit_indx].plot(state.ptrace(bqubit_indx))  # type: ignore
        else:
            for j in range(len(self.breg)):
                self.breg[j].plot(state.ptrace(j))  # type: ignore


def extend_op_to_circ(Ms: Dict[int, jnp.ndarray], bcirc: BosonicCircuit):
    """
    Arguments:
        Ms (dict):
            key: qubit index in bcirc
            value: operator corresponding to qubit
            examples:
                {0:Qobj1, 2:Qobj2}
        bcirc (BosonicCircuit):
            bosonic quantum circuit

    Returns:
        M_tot (jnp.ndarray):
            tensored operator that can act on state space of entire circuit
    """
    M_tot = None
    n = len(bcirc.breg.bqubits)
    for q_indx in range(n):
        M = Ms.get(q_indx, jqt.identity(bcirc.breg[q_indx].params["N"]))
        M_tot = M if M_tot is None else jqt.tensor(M_tot, M)
    return M_tot


class BosonicGate(metaclass=ABCMeta):
    def __init__(
        self,
        bcirc: BosonicCircuit,
        bqubit_indxs: Union[int, Tuple[int, ...]],
        params: Optional[Dict[str, Any]] = None,
        ts: Optional[jnp.ndarray] = None,
        use_unitary: Optional[bool] = False,
    ):
        if type(bqubit_indxs) == int:
            bqubit_indxs = (bqubit_indxs,)
        bqubit_indxs = cast(Tuple[int, ...], bqubit_indxs)

        self.ts = device_put(ts) if ts is not None else jnp.linspace(0, 1.0, 101)
        self.params = params if params is not None else {}
        self.params = device_put_params(self.params)

        self.args: Dict[str, complex] = {}  # used for cython qutip mesolve
        self.bcirc = bcirc
        self.bqubit_indxs = bqubit_indxs
        self.use_unitary = use_unitary

        # pre-load gates
        # self.H
        # self.U

    def __str__(self) -> str:
        return self.label

    @abstractmethod
    def get_H_func(self, t: float) -> Optional[jnp.ndarray]:
        """
        H(t), should be overriden as needed

        Args:
            t (float): time

        Returns:
            jnp.ndarray
        """

    @property
    def H_func(self):
        """
        Wrapper around get_H_func function.
        """
        if self.use_unitary:
            return None
        return self.get_H_func

    @property
    def H(self):
        """
        Allows the storage of H calculations. If use_unitary, then we forgo Hamiltonian simulation.

        Returns:
            H (list): first element is always a jnp.ndarray or 0
            other elements are lists of the form [jnp.ndarray, str]
            E.g. [sigmaz, [sigmax, "cos(t)"]]
            [0, [sigmax, "cos(t)"]]
            [sigmaz]
            [0]
        """
        if self.use_unitary:
            return None
        
        return self.get_H()

    @property
    def H_qt(self):
        if self.use_unitary:
            return None

        H = self.H
        if H is None:
            return None
        H_qt = [
            0 if isinstance(H[0], Number) and H[0] == 0 else self.bcirc.jqt2qt(H[0])
        ]
        for i in range(1, len(H)):
            H_qt.append([self.bcirc.jqt2qt(H[i][0]), H[i][1]])
        return H_qt

    @property
    def U(self):
        """
        Allows the storage of U calculations.
        """
        return self.get_U()

    @property
    def U_qt(self):
        return self.bcirc.jqt2qt(self.U)

    @property
    @abstractmethod
    def label(self) -> str:
        """
        Label of gate, used for drawing circuit.
        E.g. "X"
        """

    @abstractmethod
    def get_H(self) -> Optional[List]:
        """QuTiP cython-backend compatible Hamiltonian list.

        Returns:
            List of hamiltonians, used for simulation.
                E.g. [sigmaz, [sigmax, "cos(t)"]]
        """

    def _get_U_from_H(self) -> Optional[jnp.ndarray]:
        H = self.H
        if type(H) is list and len(H) == 1:
            H0 = H[0]
            if isinstance(H0, jnp.ndarray):
                return jsp.linalg.expm(1.0j * H0)
        return None

    def get_U(self) -> jnp.ndarray:
        U = self._get_U_from_H()
        if U is not None:
            return U
        raise NotImplementedError("Unitary gate has not been implemented.")

    def extend_gate(self, Ms: List[jnp.ndarray]):
        """
        This can be used to extend a unitary gate or a hamiltonian.
        """
        assert len(Ms) == len(self.bqubit_indxs), ValueError(
            "The number of qubit indices does not match those expected by this gate."
        )
        Ms_dict = {}
        for i, M in enumerate(Ms):
            Ms_dict[self.bqubit_indxs[i]] = M
        M_tot = extend_op_to_circ(Ms_dict, self.bcirc)
        return M_tot


def gen_custom_gate(
    Hs_func: Optional[Callable] = None,
    Us: Optional[List[jnp.ndarray]] = None,
):
    class CustomGate(BosonicGate):
        """CustomGate."""

        label = "Custom"

        def get_H(self) -> Optional[List]:
            # TODO: implement this
            return [0]

        def get_H_func(self, t: float) -> jnp.ndarray:
            if Hs_func is None:
                raise NotImplementedError(
                    "Hs_func was not provided upon initialization."
                )
            return self.extend_gate(Hs_func(t))

        def get_U(self) -> jnp.ndarray:
            if Us is None:
                raise NotImplementedError("Us was not provided upon initialization.")
            U_tot = self.extend_gate(Us)
            return U_tot

    return CustomGate


# Explicitly Referenced Gates
# ============================================================


class XGate(BosonicGate):
    label = "X"

    def get_H(self) -> Optional[List]:
        # TODO: Fix when needed
        H = self.bcirc.breg[self.bqubit_indxs[0]].x_H
        if H is None:
            return None
        Hs = [H]
        H_tot = self.extend_gate(Hs)
        return [H_tot]

    def get_H_func(self, t: float) -> jnp.ndarray:
        return self.H[0]

    def get_U(self) -> jnp.ndarray:
        Us = [self.bcirc.breg[self.bqubit_indxs[0]].x_U]
        U_tot = self.extend_gate(Us)
        return U_tot


class YGate(BosonicGate):
    label = "Y"

    def get_H(self) -> Optional[List]:
        H = self.bcirc.breg[self.bqubit_indxs[0]].y_H
        if H is None:
            return None
        Hs = [H]
        H_tot = self.extend_gate(Hs)
        return [H_tot]

    def get_H_func(self, t: float) -> jnp.ndarray:
        return self.H[0]

    def get_U(self) -> jnp.ndarray:
        Us = [self.bcirc.breg[self.bqubit_indxs[0]].y_U]
        U_tot = self.extend_gate(Us)
        return U_tot


class ZGate(BosonicGate):
    label = "Z"

    def get_H(self) -> Optional[List]:
        H = self.bcirc.breg[self.bqubit_indxs[0]].z_H
        if H is None:
            return None
        Hs = [H]
        H_tot = self.extend_gate(Hs)
        return [H_tot]

    def get_H_func(self, t: float) -> jnp.ndarray:
        return self.H[0]

    def get_U(self) -> jnp.ndarray:
        Us = [self.bcirc.breg[self.bqubit_indxs[0]].z_U]
        U_tot = self.extend_gate(Us)
        return U_tot
