# GKP Qubit Implementation

This document provides an overview of the GKP (Gottesman-Kitaev-Preskill) qubit implementation in the `bosonic` library. The GKP qubit is a type of bosonic code used for quantum error correction, and this implementation supports rectangular, square, and hexagonal GKP qubits.

---

## Overview
The GKP qubit encodes a logical qubit into the continuous-variable space of a harmonic oscillator. This implementation provides:
- A base `GKPQubit` class for general GKP qubits.
- Derived classes for specific GKP geometries: `RectangularGKPQubit`, `SquareGKPQubit`, and `HexagonalGKPQubit`.
- Common gates (`X`, `Y`, `Z`) and methods for generating basis states.

---

## Installation
To use the GKP qubit implementation, ensure you have the `bosonic` library installed. You can install it using pip:
```bash
pip install bosonic
```

## Usage
### Creating a GKP Qubit
To create a GKP qubit, initialize the GKPQubit class with the desired parameters:

```python
from bosonic.codes.gkp import GKPQubit

# Create a GKP qubit with default parameters
qubit = GKPQubit(params={"delta": 0.25, "d": 2, "N": 10})
```

#### Parameters:
- **delta**: The finite-energy parameter (default: 0.25).
- **d**: The dimension of the qudit (default: 2 for a qubit).
- **N**: The truncation level for the harmonic oscillator Hilbert space.

### Generating Basis States
The `_get_basis_z` method generates the `|+z>` and `|-z>` basis states:

```python
plus_z, minus_z = qubit._get_basis_z()
print("|+z> state:", plus_z)
print("|-z> state:", minus_z)
```

### Applying Gates
You can apply common gates like X, Y, and Z to the GKP qubit:

```python
# Apply the X gate to the |+z> state
X_gate = qubit.x_U
result = X_gate @ plus_z
print("X gate applied to |+z>:", result)
```

### Custom GKP Geometries
The library supports rectangular, square, and hexagonal GKP qubits. For example, to create a rectangular GKP qubit:

```python
from bosonic.codes.gkp import RectangularGKPQubit

rectangular_qubit = RectangularGKPQubit(params={"a": 0.8, "delta": 0.25, "d": 2, "N": 10})
```

