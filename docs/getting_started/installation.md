## Installation

*Conda users, please make sure to `conda install pip` before running any pip installation if you want to install `bosonic-jax` into your conda environment.*

#### Install from PyPI

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

#### Build from source

To build `bosonic-jax` from source, pip install using:
```
git clone git@github.com:Phionx/bosonic-jax.git
cd bosonic-jax
pip install --upgrade .
```

If you also want to download the dependencies needed to run optional tutorials, please use `pip install --upgrade .[dev,docs]` or `pip install --upgrade '.[dev,docs]'` (for `zsh` users).

***Please Note:***
For now, you will also have to manually install the `bosonic_jax` dependency, to learn how to do so please visit: [https://github.com/EQuS/bosonic-jax](https://github.com/EQuS/bosonic-jax).

#### Installation for Devs

If you intend to contribute to this project, please install `bosonic-jax` in develop mode as follows:
```sh
git clone git@github.com:Phionx/bosonic-jax.git
cd bosonic-jax
pip install -e .[dev,docs]
```
Please use `pip install -e '.[dev,docs]'` if you are a `zsh` user.


Installing the package in the usual non-editable mode would require a developer to upgrade their pip installation (i.e. run `pip install --upgrade .`) every time they update the package source code.

***Please Note:***
For now, you will also have to manually install the `bosonic_jax` dependency, to learn how to do so please visit: [https://github.com/EQuS/bosonic-jax](https://github.com/EQuS/bosonic-jax).