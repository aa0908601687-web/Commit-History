"""Cover fail-closed selection and archive auditing without weakening CI thresholds."""
import contextlib
import copy
import csv
import io
import json
import math
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import v6_selection_consistency_audit as selection

AUDIT_FILE = Path(__file__).with_name("v6_three_view_archive_audit.py")


def selection_fixture():
    scores = {str(i): 0.1 + i / 1000 for i in range(1, 39)}
    selected = [33, 34, 35, 36, 37, 38]
    best_score = sum(scores[str(n)] for n in selected)
    return {
        "openfe_feature_layer": {
            "mode": "three_history_views_100_percent",
            "full_archive_role": "training_anchors_only",
            "view_weights": dict.fromkeys(["latest_5", "latest_10", "full_archive"], 1 / 3),
            "per_view_scores": {
                name: dict(scores) for name in ("latest_5", "latest_10", "full_archive")
            },
        },
        "first_zone": selected,
        "second_zone": 1,
        "views": {"full_archive": {"rows": 20}},
        "exact_optimization_layer": {
            "exact_selection": list(selected),
            "searched_combinations": 28,
            "globally_optimal_within_pool": True,
            "exact_score": best_score,
            "approximate_selection": list(selected),
            "approximate_score": best_score,
            "score_gap": 0,
        },
    }


class SelectionConsistencyAuditTests(unittest.TestCase):
    def test_valid_fixture_and_integer_keys(self):
        data = selection_fixture()
        self.assertEqual(selection.audit(data)["status"], "PASS")
        for view_scores in data["openfe_feature_layer"]["per_view_scores"].values():
            for key in list(view_scores):
                view_scores[int(key)] = view_scores.pop(key)
        self.assertEqual(selection.audit(data)["checked_combinations"], 28)

    def test_each_failure_guard(self):
        cases = [
            lambda d: d["openfe_feature_layer"].update(mode="bad"),
            lambda d: d["openfe_feature_layer"].update(full_archive_role="bad"),
            lambda d: d["openfe_feature_layer"]["view_weights"].pop("latest_5"),
            lambda d: d["openfe_feature_layer"]["view_weights"].update(latest_5=0),
            lambda d: d["openfe_feature_layer"]["per_view_scores"]["latest_5"].pop("1"),
            lambda d: d["openfe_feature_layer"]["per_view_scores"]["latest_10"].update({"2": "invalid"}),
            lambda d: d["openfe_feature_layer"]["per_view_scores"]["full_archive"].update({"3": math.nan}),
            lambda d: d.update(first_zone=[33, 34, 35, 36, 37]),
            lambda d: d.update(first_zone=[33, 33, 35, 36, 37, 38]),
            lambda d: d.update(first_zone=[1, 34, 35, 36, 37, 38]),
            lambda d: d.update(first_zone=[32, 34, 35, 36, 37, 38]),
            lambda d: d["exact_optimization_layer"].update(exact_selection=[32, 34, 35, 36, 37, 38]),
            lambda d: d["exact_optimization_layer"].update(searched_combinations=27),
            lambda d: d["exact_optimization_layer"].update(globally_optimal_within_pool=False),
            lambda d: d["exact_optimization_layer"].update(exact_score=0),
            lambda d: d["exact_optimization_layer"].update(approximate_selection=[33, 34, 35, 36, 37]),
            lambda d: d["exact_optimization_layer"].update(approximate_selection=[33, 33, 35, 36, 37, 38]),
            lambda d: d["exact_optimization_layer"].update(approximate_selection=[1, 34, 35, 36, 37, 38]),
            lambda d: d["exact_optimization_layer"].update(approximate_score=0),
            lambda d: d["exact_optimization_layer"].update(score_gap=1),
            lambda d: d.update(second_zone=1.5),
            lambda d: d.update(second_zone=0),
            lambda d: d.update(second_zone=9),
            lambda d: d["views"]["full_archive"].update(rows=5),
        ]
        for index, mutation in enumerate(cases):
            with self.subTest(index=index):
                data = selection_fixture()
                mutation(data)
                with self.assertRaises(ValueError):
                    selection.audit(data)

    def test_cli_with_and_without_output(self):
        data = selection_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "selection.json"
            output_file = Path(tmp) / "result.json"
            input_file.write_text(json.dumps(data), encoding="utf-8")
            for args in (
                ["--result", str(input_file)],
                ["--result", str(input_file), "--output", str(output_file)],
            ):
                with self.subTest(args=args), patch.object(sys, "argv", ["audit", *args]):
                    with contextlib.redirect_stdout(io.StringIO()) as out:
                        runpy.run_module("v6_selection_consistency_audit", run_name="__main__")
                    self.assertEqual(json.loads(out.getvalue())["status"], "PASS")
            self.assertEqual(json.loads(output_file.read_text())["status"], "PASS")


class ThreeViewArchiveAuditTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [
            {"draw_id": f"1150000{i:02d}", "first": "1 2 3 4 5 6", "second": "1"}
            for i in range(70, 90)
        ]

    def run_case(self, rows, *, fieldnames=("draw_id", "first", "second"),
                 first="115000070", last="115000089", count=20, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "data.csv"
            dst = Path(tmp) / "audit.json"
            with src.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
                writer.writeheader()
                writer.writerows(rows)
            argv = [
                "archive-audit", "--csv", str(src),
                "--expected-first-draw", first,
                "--expected-last-draw", last,
                "--expected-count", str(count),
                "--output", str(dst)
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(SystemExit):
                        runpy.run_path(str(AUDIT_FILE), run_name="__main__")
                else:
                    runpy.run_path(str(AUDIT_FILE), run_name="__main__")
            if dst.exists():
                return json.loads(dst.read_text(encoding="utf-8"))

    def test_valid_and_wrong_coverage(self):
        valid = self.run_case(self.rows())
        self.assertTrue(valid["archive_coverage_verified"])
        self.assertEqual(valid["latest_5_rows"], 5)
        self.assertEqual(valid["latest_10_rows"], 10)
        for params in [{"count": 19}, {"first": "115000000"}, {"last": "115000090"}]:
            with self.subTest(params=params):
                self.assertFalse(self.run_case(self.rows(), fail=True, **params)["archive_coverage_verified"])

    def test_invalid_header(self):
        self.run_case([], fieldnames=("draw_id", "first", "wrong"), fail=True)

    def test_invalid_draw_identifiers(self):
        for mutate in (
            lambda r: r.clear(),
            lambda r: r[1].update(draw_id=r[0]["draw_id"]),
            lambda r: r.reverse(),
        ):
            with self.subTest(mutate=mutate):
                rows = self.rows()
                mutate(rows)
                self.run_case(rows, fail=True)

    def test_invalid_numbers_and_second_zone(self):
        changes = [
            {"first": "1 2 3 4 5"},
            {"first": "1 1 3 4 5 6"},
            {"first": "0 2 3 4 5 6"},
            {"first": "1 2 3 4 5 39"},
            {"second": "0"},
            {"second": "9"},
        ]
        for change in changes:
            with self.subTest(change=change):
                rows = self.rows()
                rows[0].update(change)
                self.run_case(rows, fail=True)
