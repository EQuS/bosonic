<h1 align="center">
    <img src="./docs/assets/logo_sq.png" height="120" alt="bosonic logo">
</h1>


[![License](https://img.shields.io/github/license/EQuS/bosonic.svg?style=popout-square)](https://opensource.org/license/apache-2-0) [![](https://img.shields.io/github/release/EQuS/bosonic.svg?style=popout-square)](https://github.com/EQuS/bosonic/releases) [![](https://img.shields.io/pypi/dm/bosonic.svg?style=popout-square)](https://pypi.org/project/bosonic/)

[S. R. Jha](https://github.com/Phionx), [S. Chowdhury](https://github.com/shoumikdc), [M. Hays](https://scholar.google.com/citations?user=06z0MjwAAAAJ), [J. A. Grover](https://scholar.google.com/citations?user=igewch8AAAAJ), [W. D. Oliver](https://scholar.google.com/citations?user=4vNbnqcAAAAJ&hl=en)

**Docs:** [https://equs.github.io/bosonic](https://equs.github.io/bosonic)

We present `bosonic` as a framework with which to simulate quantum circuits built using bosonic quantum-error-correctable code qubits, such as the Gottesman, Kitaev and Preskill (GKP) code. As such, we build `bosonic` ontop of `JAX` to enable the auto differentiable and (CPU, GPU, TPU) accelerated unitary and hamiltonian simulation of these quantum circuits under experimentally realisitic noise and dissipation.


## Installation

`bosonic` is published on PyPI. So, to install the latest version from PyPI, simply run the following code to install the package:

```bash
pip install bosonic
```

For more details, please visit the getting started > installation section of our [docs](https://equs.github.io/bosonic/getting_started/installation.html).


## An Example

Here's an example on how to use `bosonic`:

```python
from bosonic import BosonicRegister, GKPQubit, Qubit, BosonicCircuit, PhaseRotationGate, CDGate, execute
import jax.numpy as jnp

breg = BosonicRegister([GKPQubit(),Qubit()]) # [q0,q1]
bcirc = BosonicCircuit(breg)

bcirc.x(1) # add an X Gate on q1
bcirc.add(PhaseRotationGate, 0, {"phi": jnp.pi/4}) 
bcirc.add(CDGate, (0,1), {"beta": 1}) # q0 is the control

results = execute(bcirc, "unitary_jax")
results.plot(bcirc, 0)
results.plot(bcirc, 1)
```

## Acknowledgements & History

**Core Devs:** [Shantanu A. Jha](https://github.com/Phionx), [Shoumik Chowdhury](https://github.com/shoumikdc)


This package was initiall developed without JAX in the fall of 2021. Then, `bosonic` was rebuilt on JAX in early 2022. This package was briefly announced to the world at APS March Meeting 2023 and released to a select few academic groups shortly after. Since then, this package has been open sourced and developed while conducting research in the Engineering Quantum Systems Group at MIT with invaluable advice from [Prof. William D. Oliver](https://equs.mit.edu/william-d-oliver/). 

## Citation

Thank you for taking the time to try our package out. If you found it useful in your research, please cite us as follows:

```bibtex
@software{jha2024jaxquantum,
  author = {Shantanu R. Jha and Shoumik Chowdhury and Max Hays and Jeff A. Grover and William D. Oliver},
  title  = {An auto differentiable and hardware accelerated software toolkit for quantum circuit design, simulation and control},
  url    = {https://github.com/EQuS/jaxquantum, https://github.com/EQuS/bosonic, https://github.com/EQuS/qcsys},
  version = {0.1.0},
  year   = {2024},
}
```
> S. R. Jha, S. Chowdhury, M. Hays, J. A. Grover, W. D. Oliver. An auto differentiable and hardware accelerated software toolkit for quantum circuit design, simulation and control (2024), in preparation.


## Contributions & Contact

This package is open source and, as such, very open to contributions. Please don't hesitate to open an issue, report a bug, request a feature, or create a pull request. We are also open to deeper collaborations to create a tool that is more useful for everyone. If a discussion would be helpful, please email [shanjha@mit.edu](mailto:shanjha@mit.edu) to set up a meeting. 
