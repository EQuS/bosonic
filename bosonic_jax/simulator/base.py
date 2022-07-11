"""
Base simulator class
"""

from copy import deepcopy
from functools import partial
from numbers import Number
from typing import Optional, List
import warnings

from jaxquantum.utils.utils import is_1d
from bosonic_jax.codes.bcirc import BosonicCircuit
import jaxquantum as jqt


from jax import jit, vmap
from jax.experimental.ode import odeint
from jax import tree_util
from jax.config import config
import qutip as qt
import jax.numpy as jnp

config.update("jax_enable_x64", True)


UNITARY = "unitary"
UNITARY_JAX = "unitary_jax"
HAMILTONIAN = "hamiltonian"
HAMILTONIAN_JAX = "hamiltonian_jax"


class BosonicResults:
    """
    BosonicResults class to hold results of simulation.
    """

    def __init__(self, results: Optional[List] = None):
        self.__results = results if results is not None else []

    @property
    def results(self):
        """
        self.results is read-only
        """
        return self.__results

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.results)

    def __getitem__(self, j: int):
        return self.results[j]

    def calc_expect(self, op: jnp.ndarray, op_name: str):
        """
        TODO (if needed): 
        This only works for jax simulations. 
        We may want to extend it to accomodate QuTiP mesolve sims.
        """

        @jit
        def calc_exp_state(state: jnp.ndarray):
            return (jqt.dag(state) @ op @ state)[0][0]

        for i in range(len(self.results)):
            if op_name not in self.results[i]:
                self.results[i][op_name] = vmap(calc_exp_state)(
                    self.results[i]["states"]
                )

    def append(self, states):
        self.__results.append({"states": states})

    def plot(self, bcirc: BosonicCircuit, indx: Optional[int] = None):
        """
        Plots final state.
        """
        bcirc.plot(indx, state=self.results[-1]["states"][-1])


# This allows us to return BosonicResults from a jitted function.
tree_util.register_pytree_node(
    BosonicResults,
    lambda res: ((res.results,), None),
    lambda _, xres: BosonicResults(xres[0]),
)


def execute(bcirc: BosonicCircuit, backend: str, **kwargs):
    p0 = kwargs.pop("p0", None)
    H0 = kwargs.pop("H0", 0)
    if backend == UNITARY:
        warnings.warn(
            "Please use the 'unitary_jax' backend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return unitary_simulate(bcirc, p0=p0)
    elif backend == UNITARY_JAX:
        return unitary_jax_simulate(bcirc, p0=p0)
    elif backend == HAMILTONIAN:
        warnings.warn(
            "Please use the 'hamiltonian_jax' backend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return hamiltonian_simulate(bcirc, H0, p0=p0, **kwargs)
    elif backend == HAMILTONIAN_JAX:
        return hamiltonian_jax_simulate(bcirc, H0, p0=p0, **kwargs)


def unitary_step(psi, U):
    return (U * psi).unit() if qt.isket(psi) else (U * psi * U.dag()).unit()


def unitary_simulate(bcirc: BosonicCircuit, p0=None):
    p = p0 if p0 is not None else bcirc.default_initial_state.copy()
    p = bcirc.jax2qt(p)

    results = BosonicResults()
    for gate in bcirc.gates:
        U = gate.U_qt
        p = unitary_step(p, U)
        results.append([p])
    return results


def hamiltonian_simulate(
    bcirc: BosonicCircuit, H0, p0=None, default_unitary=True, **kwargs
):
    kwargs["options"] = kwargs.get("options", qt.Options(store_states=True))

    # p0 is a density matrix, but can also be wavefunction if c_ops=None
    p = p0 if p0 is not None else bcirc.default_initial_state.copy()

    p = bcirc.jax2qt(p)
    H0 = bcirc.jax2qt(H0)

    results = BosonicResults()
    for gate in bcirc.gates:
        t_list = gate.ts
        H = deepcopy(gate.H_qt)
        args = gate.args
        if (
            H
            and type(H) is list
            and (type(H[0]) is qt.Qobj or (isinstance(H[0], Number) and H[0] == 0))
        ):
            H[0] += H0
            result = qt.mesolve(H, p, t_list, args=args, **kwargs)
            results.append(result.states)
            p = results[-1]["states"][-1]
        elif (
            default_unitary
        ):  # if H is [] or None for a gate, then just use unitary evolution
            U = gate.U_qt
            p = unitary_step(p, U)
            results.append([p])
        else:
            warnings.warn(f"{gate} Gate was skipped.", Warning, stacklevel=2)

    return results


# JAX
# ==================================================================================
def spre(op):
    op_dag = jnp.conj(op.T)
    return lambda rho: 0.5 * (
        2 * op @ rho @ op_dag - rho @ op_dag @ op - op_dag @ op @ rho
    )


@partial(jit, static_argnums=(0,))
def unitary_jax_simulate(bcirc: BosonicCircuit, p0=None):
    p = p0 if p0 is not None else bcirc.default_initial_state.copy()
    p = jqt.qt2jax(p)
    use_density_matrix = not is_1d(p)
    results = BosonicResults()
    for gate in bcirc.gates:
        U = jqt.qt2jax(gate.U)
        p = unitary_jax_step(p, U, use_density_matrix=use_density_matrix)
        results.append([p])
    return results


@partial(jit, static_argnums=(0, 5,))
def hamiltonian_jax_step(
    H_func, # H_func stores gate dynamics
    p: jnp.ndarray,
    t_list: jnp.ndarray,
    H0: jnp.ndarray, # H0 represents the base system dynamics
    c_ops=None,
    use_density_matrix=False,
):
    @jit
    def Ht(t):
        return H_func(t) + H0

    return jqt.mesolve(p, t_list, c_ops=c_ops, Ht=Ht, use_density_matrix=use_density_matrix)

@partial(jit, static_argnums=(2,))
def unitary_jax_step(rho, U, use_density_matrix=False):
    if use_density_matrix:
        U_dag = jnp.conj(U).T
        return U @ rho @ U_dag
    return U @ rho


@partial(jit, static_argnums=(0, 3))
def hamiltonian_jax_simulate(
    bcirc: BosonicCircuit,
    H0: jnp.ndarray,
    p0: jnp.ndarray = None,
    default_unitary=True,
    c_ops=None,
    results_in: Optional[BosonicResults] = None,
):
    """
    
    Args:
        H0 (jnp.ndarray):
            base system hamiltonian, 
            please make sure this a jnp.array not a QuTiP Qobj
    """
    # p0 is a density matrix, but can also be wavefunction if c_ops=None
    p = p0 if p0 is not None else bcirc.default_initial_state.copy()
    c_ops = c_ops if c_ops is not None else []

    p = jqt.qt2jax(p)

    if c_ops != [] and is_1d(p):
        # if simulating with noise and p is a vector,
        # then turn p into a density matrix
        p = p @ jnp.conj(p).T

    use_density_matrix = not is_1d(p)
    results = BosonicResults() if results_in is None else results_in

    for gate in bcirc.gates:
        t_list = gate.ts
        use_hamiltonian = not (gate.H_func is None or gate.H_func(t_list[0]) is None)

        if use_hamiltonian:
            states = hamiltonian_jax_step(
                gate.H_func,
                p,
                t_list,
                H0,
                c_ops=c_ops,
                use_density_matrix=use_density_matrix,
            )
            results.append(states)
            p = states[-1]
        elif default_unitary:
            # H_func is None or returns None, then just use unitary evolution
            U = jqt.qt2jax(gate.U)
            p = unitary_jax_step(p, U, use_density_matrix=use_density_matrix)
            results.append(jnp.array([p]))
        else:
            warnings.warn(f"{gate} Gate was skipped.", RuntimeWarning, stacklevel=2)

    return results

