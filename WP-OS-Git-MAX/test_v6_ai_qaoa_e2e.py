import csv
import importlib.util
import tempfile
import types
import subprocess
import sys
import json
import runpy
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location("v6_ai_qaoa_e2e", HERE / "v6_ai_qaoa_e2e.py")
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)

class V6ValidationTests(unittest.TestCase):
    def make_csv(self, rows=12):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "draws.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["draw_id", "first", "second"])
            for i in range(rows):
                first = sorted(((i*3+j*5) % 38)+1 for j in range(6))
                w.writerow([f"115{(i+1):06d}", " ".join(f"{x:02d}" for x in first), f"{i%8+1:02d}"])
        return tmp, path

    def test_three_views_and_output(self):
        tmp, path = self.make_csv()
        with tmp:
            result = v6.run(path)
            self.assertEqual(result["views"]["latest_5"]["rows"], 5)
            self.assertEqual(result["views"]["latest_10"]["rows"], 10)
            self.assertEqual(result["views"]["complete_archive"]["rows"], 12)
            self.assertEqual(len(result["first_zone"]), 6)
            self.assertEqual(len(set(result["first_zone"])), 6)
            self.assertTrue(all(1 <= n <= 38 for n in result["first_zone"]))
            self.assertIn(result["second_zone"], range(1, 9))
            self.assertEqual(v6.run(path), result)

    def test_high_fidelity_e2e_is_local_and_not_hit_rate(self):
        tmp,path=self.make_csv()
        with tmp:
            result=v6.run(path,high_fidelity_verify=True)
            layer=result["high_fidelity_layer"]
            self.assertFalse(layer["enabled"])
            self.assertIn("removed", layer["scope"])
            self.assertNotIn("qpu", layer)
            self.assertNotIn("jupiter_nvidia_connected", layer)
            self.assertNotIn("meaning", layer)

    def test_cudaq_result_replaces_local_candidate(self):
        tmp, path = self.make_csv()
        fake = types.SimpleNamespace(
            select_six=lambda pool, scores, required=False: (
                sorted(pool[:6]),
                {"enabled": True, "executed": True, "qpu": False, "required": required},
            )
        )
        with tmp, patch.dict(sys.modules, {"v6_cudaq_backend": fake}):
            result = v6.run(path, use_cudaq=True)
            required_result = v6.run(path, require_cudaq=True)
        fallback = types.SimpleNamespace(
            select_six=lambda pool, scores, required=False: (
                None, {"enabled": True, "executed": False, "qpu": False}
            )
        )
        fallback_tmp, fallback_path = self.make_csv()
        with fallback_tmp, patch.dict(sys.modules, {"v6_cudaq_backend": fallback}):
            fallback_result = v6.run(fallback_path, use_cudaq=True)
        self.assertFalse(fallback_result["cudaq_layer"]["executed"])
        self.assertTrue(required_result["cudaq_layer"]["required"])
        self.assertTrue(result["cudaq_layer"]["executed"])
        self.assertFalse(result["cudaq_layer"]["qpu"])
        self.assertEqual(len(result["first_zone"]), 6)

    def test_main_in_process_for_coverage(self):
        tmp, path = self.make_csv()
        with tmp:
            out = Path(tmp.name) / "result.json"
            buf = io.StringIO()
            with patch.object(sys, "argv", ["v6", "--csv", str(path), "--output", str(out)]), redirect_stdout(buf):
                v6.main()
            self.assertEqual(json.loads(buf.getvalue()), json.loads(out.read_text()))
            with patch.object(sys, "argv", ["v6", "--csv", str(path)]), redirect_stdout(buf):
                v6.main()
            self.assertTrue(buf.getvalue())

    def test_cli_output_and_stdout(self):
        tmp, path = self.make_csv()
        with tmp:
            output = Path(tmp.name) / "output.json"
            proc = subprocess.run(
                [sys.executable, str(HERE / "v6_ai_qaoa_e2e.py"), "--csv", str(path), "--output", str(output)],
                check=True, capture_output=True, text=True
            )
            self.assertEqual(json.loads(proc.stdout), json.loads(output.read_text()))
            self.assertEqual(json.loads(proc.stdout)["last_draw"], "115000012")

    def test_cli_without_output(self):
        tmp, path = self.make_csv()
        with tmp:
            proc = subprocess.run(
                [sys.executable, str(HERE / "v6_ai_qaoa_e2e.py"), "--csv", str(path)],
                check=True, capture_output=True, text=True
            )
            self.assertEqual(json.loads(proc.stdout)["views"]["latest_10"]["rows"], 10)


    def test_rejects_invalid_zone_two_and_zone_one(self):
        tmp,path=self.make_csv()
        with tmp:
            original=path.read_text()
            lines=original.splitlines()
            columns=lines[1].split(",")
            columns[2]="09"
            lines[1]=",".join(columns)
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaises(ValueError): v6.validate(path)
            lines=original.splitlines()
            columns=lines[1].split(",")
            columns[1]="01 06 11 16 21 39"
            lines[1]=",".join(columns)
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaises(ValueError): v6.validate(path)

    def test_rejects_invalid_period_and_count(self):
        tmp,path=self.make_csv()
        with tmp:
            original=path.read_text()
            lines=original.splitlines()
            columns=lines[1].split(",")
            columns[0]="BAD"
            lines[1]=",".join(columns)
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaises(ValueError): v6.validate(path)
            lines=original.splitlines()
            columns=lines[1].split(",")
            columns[1]="01 06 11 16 21"
            lines[1]=",".join(columns)
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaises(ValueError): v6.validate(path)

    def test_script_entrypoint_in_coverage(self):
        tmp, path = self.make_csv()
        with tmp:
            buf = io.StringIO()
            with patch.object(sys, "argv", ["v6", "--csv", str(path)]), redirect_stdout(buf):
                runpy.run_path(str(HERE / "v6_ai_qaoa_e2e.py"), run_name="__main__")
            self.assertEqual(json.loads(buf.getvalue())["last_draw"], "115000012")

    def test_rejects_too_few_rows(self):
        tmp,path=self.make_csv(9)
        with tmp, self.assertRaises(ValueError): v6.validate(path)

    def test_rejects_duplicate_period(self):
        tmp,path=self.make_csv()
        with tmp:
            lines=path.read_text().splitlines()
            path.write_text("\n".join(lines+[lines[1]])+"\n")
            with self.assertRaises(ValueError): v6.validate(path)

    def test_rejects_invalid_number(self):
        tmp,path=self.make_csv()
        with tmp:
            content=path.read_text().replace("01 06 11 16 21 26", "01 01 11 16 21 26")
            path.write_text(content)
            with self.assertRaises(ValueError): v6.validate(path)

    def test_rejects_wrong_header(self):
        tmp,path=self.make_csv()
        with tmp:
            path.write_text(path.read_text().replace("draw_id,first,second","id,first,second"))
            with self.assertRaises(ValueError): v6.validate(path)

    def test_rejects_out_of_order(self):
        tmp,path=self.make_csv()
        with tmp:
            lines=path.read_text().splitlines()
            lines[1],lines[2]=lines[2],lines[1]
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaises(ValueError): v6.validate(path)

if __name__=="__main__":
    unittest.main()
