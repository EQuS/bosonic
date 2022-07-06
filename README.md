# bosonic-jax

[![License](https://img.shields.io/github/license/Phionx/bosonic-jax.svg?style=popout-square)](https://opensource.org/licenses/MIT) [![](https://img.shields.io/github/release/Phionx/bosonic-jax.svg?style=popout-square)](https://github.com/Phionx/bosonic-jax/releases) [![](https://img.shields.io/pypi/dm/bosonic-jax.svg?style=popout-square)](https://pypi.org/project/bosonic-jax/)

***Simulating quantum circuits built using bosonic QEC code qubits with JAX.***

## Motivation

We present `bosonic-jax` as a framework with which to simulate quantum circuits built using bosonic quantum-error-correctable code qubits, such as the Gottesman, Kitaev and Preskill (GKP) code. As such, we build `bosonic-jax` ontop of `JAX` to enable the accelerated unitary and hamiltonian simulation of these quantum circuits with experimentally realisitic noise and dissipation.


## Installation

*Conda users, please make sure to `conda install pip` before running any pip installation if you want to install `bosonic-jax` into your conda environment.*

`bosonic-jax` will soon be published on PyPI. So, to install, simply run:

```python
pip install bosonic-jax
```


To check if the installation was successful, run:
```python
python3
>>> import bosonic_jax as bcj
```

If pip installation doesn't work, please build from source, as detailed below. 

#### Building from source

To build `bosonic-jax` from source, pip install using:
```
git clone git@github.com:Phionx/bosonic-jax.git
cd bosonic-jax
pip install --upgrade .
```


If you also want to download the dependencies needed to run optional tutorials, please use `pip install --upgrade .[dev]` or `pip install --upgrade '.[dev]'` (for `zsh` users).

#### Installation for Devs

If you intend to contribute to this project, please install `bosonic-jax` in develop mode as follows:
```sh
git clone git@github.com:Phionx/bosonic-jax.git
cd bosonic-jax
pip install -e .[dev]
```
Please use `pip install -e '.[dev]'` if you are a `zsh` user.


Installing the package in the usual non-editable mode would require a developer to upgrade their pip installation (i.e. run `pip install --upgrade .`) every time they update the package source code.

#### Building documentation for Devs

Set yourself up to use the `[dev]` dependencies. Then, from the command line run:
```bash
mkdocs build
```

Then, when you're ready to deploy, run:
```bash
mkdocs gh-deploy
```
## Codebase

The codebase is split across `bosonic_jax/codes` and `bosonic_jax/simulator`, which respectively provide tooling for several bosonic QEC codes (e.g. cat, binomial, GKP codes) and simulators with which to benchmark circuits built using these code qubits.

## Future Directions

Checkout [issues](https://github.com/Phionx/bosonic-jax/issues) to see what we are working on these days!

## Acknowledgements 

**Core Devs:** [Shantanu Jha](https://github.com/Phionx), [Shoumik Chowdhury](https://github.com/shoumikdc), [Max Hays](https://scholar.google.com/citations?user=06z0MjwAAAAJ&hl=en)

This package was developed while conducting research in the Engineering Quantum Systems Group at MIT with invaluable advice from [Prof. William D. Oliver](https://equs.mit.edu/william-d-oliver/).