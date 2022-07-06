"""
Stabilization Class
TODO: convert to JAX
"""

from functools import partial
from typing import Optional, Dict

from bosonic_jax.codes.bcirc import BosonicCircuit
from bosonic_jax.simulator.base import execute
import bosonic_jax.jax_qutip as jqt


from jax import jit, vmap
from jax.lax import scan
import jax.numpy as jnp


@partial(jit, static_argnums=(0, 1, 2, 3))
def sBs_stabilize(
    bcirc_x: BosonicCircuit,
    bcirc_p: BosonicCircuit,
    backend: str,
    N_rounds: int,
    p0=None,
    H0=0,
    meas_ops: Optional[jnp.ndarray] = None,
    resource_states: Optional[Dict[str, jnp.ndarray]] = None,
    c_ops: Optional[jnp.ndarray] = None,
):
    """
    Function to implement the sBs GKP stabilization protocol, as described in the reference:
    B. Royer, et. al. "Stabilization of Finite-Energy Gottesman-Kitaev-Preskill States" (2020).

    Arguments:
        bcirc_x: BosonicCircuit for single run of sBs to stabilize in X
        bcirc_p: BosonicCircuit for single run of sBs to stabilize in P
        N_rounds: Number of stabilization rounds. Each round consists of 2 sBs runs (both X and P)
        backend: Simulator backend: "unitary" or "hamiltonian"
        p0: Initial cavity state to stabilize.
        meas_ops (jnp.ndarray):
                List (jnp.ndarray) of phase-estimation measurement operators [Mx, My, Mz].
                If empty or None, then we will measure the logicals of GKP.
                TODO:
                    We should switch to a meas_circ and meas_op approach,
                    so that we can account for measurement loss.

    Returns:
        meas_results: Array of expectation values with dimension len(meas_ops) x len(N_rounds)
    
    Note: Here we assume that the GKP Qubit being stabilized has idx 0. 
    This allows us to use rho.ptrace(0) to extract just the GKP state.
    """

    # One-Time SETUP
    # --------------------------------------------------------------------
    GKP = bcirc_x.breg.bqubits[0]
    control = bcirc_x.breg.bqubits[1]
    trace_dims = (
        GKP.params["N"],
        control.params["N"],
    )  # for some reason using ,dims is troublesome

    c_ops = jnp.array([]) if c_ops is None else c_ops
    resource_states = {} if resource_states is None else resource_states

    single_qubit_g = resource_states.get(
        "single_qubit_g", jqt.ket2dm(jqt.basis(2, 0))
    )  # Qubit |g><g|

    plus = resource_states.get(
        "plus", jqt.ket2dm(1 / jnp.sqrt(2) * (jqt.basis(2, 0) + jqt.basis(2, 1)))
    )  # Qubit |+><+|

    resource_states = {} if resource_states is None else resource_states

    measuring_logicals = (
        meas_ops is None or len(meas_ops) == 0
    )  # measure logicals if meas_ops empty
    meas_ops = (
        meas_ops if meas_ops is not None else jnp.array([GKP.x_U, GKP.y_U, GKP.z_U])
    )

    rho = p0 if p0 is not None else bcirc_x.default_initial_state

    # --------------------------------------------------------------------

    @jit
    def sBs_round_func(rho_i, _):
        meas_result = sBs_stabilize_meas(
            rho_i, trace_dims, meas_ops, measuring_logicals, single_qubit_g
        )
        rho_i = sBs_stabilize_circ(
            bcirc_x, bcirc_p, trace_dims, backend, rho_i, H0, plus, c_ops
        )
        return rho_i, meas_result

    _, meas_results = scan(sBs_round_func, rho, None, length=N_rounds)
    return meas_results


@partial(jit, static_argnums=(1, 3,))
def sBs_stabilize_meas(
    rho: jnp.ndarray,
    trace_dims: tuple,
    meas_ops: jnp.ndarray,
    measuring_logicals: bool,
    single_qubit_g: jnp.ndarray,
):
    # print(trace_dims)
    N = trace_dims[0]

    # Measure first in the initial state before stabilizing
    @jit
    def measure_logicals(M):
        return jqt.tr(M @ jqt.ptrace(rho, 0, trace_dims))

    @jit
    def projective_measure(M):
        psi_meas = (
            M @ jqt.tensor(jqt.ptrace(rho, 0, trace_dims), single_qubit_g) @ jqt.dag(M)
        )
        return jqt.tr(psi_meas @ jqt.tensor(jqt.identity(N), jqt.sigmaz()))

    if measuring_logicals:
        return vmap(measure_logicals)(meas_ops)
    else:
        return vmap(projective_measure)(meas_ops)


@partial(jit, static_argnums=(0, 1, 2, 3,))
def sBs_stabilize_circ(
    bcirc_x: BosonicCircuit,
    bcirc_p: BosonicCircuit,
    trace_dims: tuple,
    backend: str,
    rho: jnp.ndarray,
    H0: jnp.ndarray,
    plus: jnp.ndarray,
    c_ops: jnp.ndarray,
):
    # Initialize ancilla to |+>, then stabilize X
    results = execute(
        bcirc_x,
        backend=backend,
        p0=jqt.tensor(jqt.ptrace(rho, 0, trace_dims), plus),
        H0=H0,
        c_ops=c_ops,
    )
    rho = results[-1]["states"][-1]

    # Reset ancilla to |+>, then stabilize P
    results = execute(
        bcirc_p,
        backend=backend,
        p0=jqt.tensor(jqt.ptrace(rho, 0, trace_dims), plus),
        H0=H0,
        c_ops=c_ops,
    )
    rho = results[-1]["states"][-1]
    return rho
