"""Standalone OF-LGBM dependency and workflow isolation contract checks.

Run without importing the model: python -m unittest verification/test_of_lgbm_isolation.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verification" / "of_lgbm_all_operator_e2e.py"
REQUIREMENTS = ROOT / "verification" / "of_lgbm_requirements.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "of-lgbm-23-operator-e2e.yml"

# Only the standalone project dependencies should be installed.
ALLOWED_DEPS = {"openfe", "lightgbm", "numpy", "pandas", "scipy", "matplotlib", "pyarrow"}
FORBIDDEN_IMPORTS = {
    "cudaq", "dimod", "qiskit", "pennylane", "torch",
    "verification.openfe_three_lottery_integration",
    "verification.v6_ising_qubo_reference",
}


class StandaloneOFLCGBMContracts(unittest.TestCase):
    def test_only_allowed_third_party_install_requirements(self):
        lines = [
            l.strip() for l in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")
        ]
        self.assertEqual(
            {re.split(r"[<>=!~\[\] ]", l, maxsplit=1)[0].lower() for l in lines},
            ALLOWED_DEPS,
        )
        self.assertEqual(len(lines), len(ALLOWED_DEPS))

    def test_runner_does_not_import_retired_or_quantum_modules(self):
        module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        imports = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        for name in imports:
            for forbidden in FORBIDDEN_IMPORTS:
                self.assertFalse(
                    name == forbidden or name.startswith(forbidden + "."),
                    msg=f"Non-standalone import detected: {name}",
                )

    def test_workflow_uses_only_standalone_runner(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pip install -r verification/of_lgbm_requirements.txt", workflow)
        self.assertIn("verification/of_lgbm_all_operator_e2e.py", workflow)
        for forbidden_script in (
            "verification/openfe_three_lottery_integration.py",
            "verification/v6_ising_qubo_reference.py",
        ):
            self.assertNotIn(forbidden_script, workflow)
        self.assertIn("verification/test_of_lgbm_isolation.py", workflow)

    def test_python_exact_search_without_external_solver(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("from itertools import combinations", source)
        self.assertIn("tickets = list(combinations(pool, count))", source)


if __name__ == "__main__":
    unittest.main()
