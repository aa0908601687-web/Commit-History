"""Offline contract checks for the official Power Lottery API adapter.

These tests do not assert that the live official endpoint is reachable.
"""
import io
import json
import unittest
from unittest.mock import patch

import update_official_data as updater


def payload(rows):
    return {"content": {"superLotto638Res": rows}}


def row(period="115000088", numbers=None):
    return {"period": period, "drawNumberSize": numbers or [1, 5, 9, 12, 20, 38, 4]}


class OfficialAPIContractTests(unittest.TestCase):
    def test_parses_valid_draw_and_sorts_first_zone(self):
        result = updater.parse_response(payload([row(numbers=[38, 1, 12, 5, 20, 9, 4])]))
        self.assertEqual(result, [("115000088", [1, 5, 9, 12, 20, 38], 4)])

    def test_rejects_missing_or_changed_response_schema(self):
        for value in ({}, {"content": None}, {"content": {}},
                      {"content": {"superLotto638Res": {}}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                updater.parse_response(value)

    def test_rejects_empty_response(self):
        with self.assertRaisesRegex(ValueError, "No official draws"):
            updater.parse_response(payload([]))

    def test_rejects_bad_period_or_number_count(self):
        for bad in (row(period="123"), row(period="abcdefghi"),
                    row(numbers=[1, 2, 3, 4, 5, 6])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                updater.parse_response(payload([bad]))

    def test_rejects_duplicate_and_out_of_range_numbers(self):
        for numbers in ([1, 1, 3, 4, 5, 6, 2],
                        [0, 2, 3, 4, 5, 6, 2],
                        [1, 2, 3, 4, 5, 39, 2],
                        [1, 2, 3, 4, 5, 6, 9]):
            with self.subTest(numbers=numbers), self.assertRaises(ValueError):
                updater.parse_response(payload([row(numbers=numbers)]))

    def test_rejects_duplicate_draw_periods(self):
        with self.assertRaisesRegex(ValueError, "Duplicate official periods"):
            updater.parse_response(payload([row(), row()]))

    def test_fetch_uses_official_endpoint_and_parses_json(self):
        body = json.dumps(payload([row()])).encode("utf-8")
        with patch.object(updater, "urlopen", return_value=io.BytesIO(body)) as mock:
            result = updater.fetch("2026-11")
        request = mock.call_args.args[0]
        self.assertTrue(request.full_url.startswith(updater.URL + "?"))
        self.assertIn("month=2026-11", request.full_url)
        self.assertEqual(mock.call_args.kwargs["timeout"], 25)
        self.assertEqual(result[0][0], "115000088")


if __name__ == "__main__":
    unittest.main()
