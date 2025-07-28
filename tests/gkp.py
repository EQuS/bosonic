import unittest
import jax.numpy as jnp
import jaxquantum as jqt
from bosonic.codes.gkp import GKPQubit
class TestGKPQubit(unittest.TestCase):

    def setUp(self):
        # Initialize GKPQubit with default parameters
        self.qubit = GKPQubit(params={"N": 10})

    def test_params_validation(self):
        # Test if default parameters are set correctly
        self.assertEqual(self.qubit.params["delta"], 0.25)
        self.assertEqual(self.qubit.params["d"], 2)
        self.assertAlmostEqual(self.qubit.params["l"], jnp.sqrt(jnp.pi * 2))
        self.assertAlmostEqual(self.qubit.params["epsilon"], jnp.sinh(0.25**2) * jnp.sqrt(jnp.pi * 2))

    def test_common_gates(self):
        # Test if common gates are generated correctly
        self.assertIn("x", self.qubit.common_gates)
        self.assertIn("p", self.qubit.common_gates)
        self.assertIn("E", self.qubit.common_gates)
        self.assertIn("E_inv", self.qubit.common_gates)
        self.assertIn("X", self.qubit.common_gates)
        self.assertIn("Z", self.qubit.common_gates)
        self.assertIn("Y", self.qubit.common_gates)
        self.assertIn("Z_s_0", self.qubit.common_gates)
        self.assertIn("S_x_0", self.qubit.common_gates)
        self.assertIn("S_z_0", self.qubit.common_gates)
        self.assertIn("S_y_0", self.qubit.common_gates)

    def test_basis_z(self):
        plus_z, minus_z = self.qubit._get_basis_z()
        self.assertIsInstance(plus_z, jqt.Qarray)
        self.assertIsInstance(minus_z, jqt.Qarray)
        # Check the shape of the underlying data
        self.assertEqual(plus_z.data.shape, (self.qubit.params["N"], 1))
        self.assertEqual(minus_z.data.shape, (self.qubit.params["N"], 1))

    def test_gates(self):
        # Test if gate properties return correct types
        self.assertIsInstance(self.qubit.x_U, jqt.Qarray)
        self.assertIsInstance(self.qubit.y_U, jqt.Qarray)
        self.assertIsInstance(self.qubit.z_U, jqt.Qarray)

    def test_make_op_finite_energy(self):
        # Test if finite energy operation is applied correctly
        op = self.qubit.common_gates["X"]
        finite_op = self.qubit._make_op_finite_energy(op)
        self.assertIsInstance(finite_op, jqt.Qarray)

    def test_symmetrized_expm(self):
        # Test if symmetrized exponential is applied correctly
        op = self.qubit.common_gates["x"]
        symm_op = self.qubit._symmetrized_expm(op)
        self.assertIsInstance(symm_op, jqt.Qarray)

if __name__ == "__main__":
    unittest.main()