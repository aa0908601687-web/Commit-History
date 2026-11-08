"""Production V6 CPU Ising/QUBO parity integration tests."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "WP-OS-Git-MAX"))
sys.path.insert(0, str(ROOT / "verification"))
from v6_ai_qaoa_e2e import run
from v6_ising_qubo_reference import check_solver_output

class TestIsingProduction(unittest.TestCase):
    def test_integration_with_real_archive(self):
        csv_path = ROOT / "WP-OS-Git-MAX" / "historical_archive.csv"
        result = run(str(csv_path), ising_qubo_check=True)
        layer = result["ising_qubo_layer"]
        self.assertTrue(layer["executed"])
        self.assertTrue(layer["same_optimal_score"])
        self.assertEqual(layer["combinations_checked"], 28)
        self.assertEqual(layer["pool"], result["candidate_pool"])
        self.assertEqual(result["first_zone"], result["exact_optimization_layer"]["exact_selection"])
        self.assertFalse(layer["nvidia_native_solver_executed"])
        self.assertFalse(layer["rocq_runtime_proof"])

    def test_fail_closed_when_qubo_reference_is_inconsistent(self):
        # Exercise the production fail-closed branch: a corrupted/inconsistent
        # QUBO reference must raise instead of silently selecting numbers.
        csv_path = ROOT / "WP-OS-Git-MAX" / "historical_archive.csv"
        with patch("v6_ising_qubo_reference.exact_six",
                   return_value={"indices": (0, 1, 2, 3, 4, 5),
                                 "energy": 0.0, "combinations_checked": 27}):
            with self.assertRaisesRegex(RuntimeError, "Ising/QUBO CPU reference disagrees"):
                run(str(csv_path), ising_qubo_check=True)

    def test_flag_off_does_not_execute(self):
        csv_path = ROOT / "WP-OS-Git-MAX" / "historical_archive.csv"
        result = run(str(csv_path))
        self.assertFalse(result["ising_qubo_layer"]["executed"])

    def test_invalid_six_of_eight_rejected(self):
        check = check_solver_output([1] * 8, [float(n) for n in range(8)])
        self.assertFalse(check["valid"])
        self.assertFalse(check["optimal"])

if __name__ == "__main__":
    unittest.main()
