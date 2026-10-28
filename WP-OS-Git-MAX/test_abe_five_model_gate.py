import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abe_gate", ROOT / "abe_five_model_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FiveModelAcceptanceTests(unittest.TestCase):
    def test_missing_archive_fails_closed(self):
        with patch.object(gate, "CSV", Path("/nonexistent/abe-test-history.csv")):
            with self.assertRaisesRegex(SystemExit, "historical archive missing"):
                gate.main()

    def test_insufficient_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["draw_id", "first", "second"])
                writer.writeheader()
                for i in range(19):
                    writer.writerow({"draw_id": str(i + 1), "first": "01 02 03 04 05 06", "second": 1})
            with patch.object(gate, "CSV", path):
                with self.assertRaisesRegex(SystemExit, "requires >=20"):
                    gate.main()

    def test_missing_lstm_command_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["draw_id", "first", "second"])
                writer.writeheader()
                for i in range(20):
                    writer.writerow({"draw_id": str(i + 1), "first": "01 02 03 04 05 06", "second": 1})
            response = '{"through_draw":"20","first":[1,2,3,4,5,6],"second":1}'
            with (patch.object(gate, "CSV", path),
                  patch.dict(gate.MODELS, {"bayesian": ["python"], "lstm": ""}),
                  patch.object(gate.subprocess, "run", return_value=type("Result", (), {"stdout": response})())):
                with self.assertRaisesRegex(SystemExit, "lstm inference command not configured"):
                    gate.main()


if __name__ == "__main__":
    unittest.main()
