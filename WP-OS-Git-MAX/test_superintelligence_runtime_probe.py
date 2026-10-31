import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("superintelligence_runtime_probe.py")
SPEC = importlib.util.spec_from_file_location("superintelligence_runtime_probe", MODULE_PATH)
runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime)


class RuntimeProbeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        agent = root / "agents" / "base_agent.py"
        learning = root / "auto_improvement" / "adaptive_learning.py"
        agent.parent.mkdir(parents=True)
        learning.parent.mkdir(parents=True)
        agent.write_text(
            "class BaseAgent:\n"
            "    def __init__(self, model, agent_id):\n"
            "        self.model=model; self.id=agent_id\n"
            "    def perform_task(self, data):\n"
            "        raise NotImplementedError\n",
            encoding="utf-8",
        )
        learning.write_text(
            "class AdaptiveLearning:\n"
            "    def __init__(self, model): self.model=model\n"
            "    def continuous_learning(self, data, labels):\n"
            "        print('Continuous learning process initiated.')\n"
            "        self.model.fit(data, labels, epochs=5)\n"
            "        print('Model has adapted with new data.')\n",
            encoding="utf-8",
        )

    def test_probe_executes_selected_upstream_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            result = runtime.probe(root)
            self.assertTrue(result["runtime_probe_passed"])
            self.assertFalse(result["upstream_main_executed"])
            self.assertFalse(result["prediction_integration"])
            self.assertEqual(result["fit_call"]["epochs"], 5)
            self.assertEqual(len(result["modules_executed"]), 2)

    def test_cli_writes_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            self._fixture(root)
            argv = ["probe", "--upstream-dir", str(root), "--output", str(output)]
            with (
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                code = runtime.main()
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["runtime_probe_passed"])
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                json.loads(stdout.getvalue()),
            )

    def test_missing_module_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                runtime.probe(Path(directory))


if __name__ == "__main__":
    unittest.main()
