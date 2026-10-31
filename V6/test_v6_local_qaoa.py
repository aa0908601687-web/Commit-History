"""Independent unit checks for V6 local simulation (not lottery validation)."""
import unittest
from v6_local_qaoa import solve

class V6Tests(unittest.TestCase):
    def test_cardinality_and_norm(self):
        out=solve([1,2,3,4,5,6,7,8],6)
        self.assertEqual(len(out["selected_indices"]),6)
        self.assertEqual(len(set(out["selected_indices"])),6)
        self.assertAlmostEqual(out["total_probability"],1.0,places=10)
        self.assertFalse(out["real_qpu"])
        self.assertFalse(out["predictive_advantage_verified"])
    def test_deterministic(self):
        self.assertEqual(solve([1,2,3,4,5,6,7,8]),solve([1,2,3,4,5,6,7,8]))
    def test_invalid(self):
        with self.assertRaises(ValueError): solve([1,2],3)
        with self.assertRaises(ValueError): solve([float("nan"),2],1)

if __name__=="__main__":
    unittest.main()
