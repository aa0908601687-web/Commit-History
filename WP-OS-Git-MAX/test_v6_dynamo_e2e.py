"""Offline Dynamo contract and mock transport coverage (no network access)."""
import io
import json
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import v6_dynamo_e2e as d


class DynamoContracts(unittest.TestCase):
    def test_payload_and_validate(self):
        self.assertEqual(d.payload("model-A")["model"], "model-A")
        self.assertTrue(d.validate({"choices": [{"message": {"content": "OK"}}]}))
        for response in (None, {}, {"choices": []},
                         {"choices": "oops"}, {"choices": [{"message": {"content": 4}}]}):
            with self.subTest(response=response), self.assertRaises((AssertionError, AttributeError)):
                d.validate(response)

    def test_url_guard(self):
        with patch.dict("os.environ", {"DYNAMO_URL": "file:///tmp/x"}, clear=True):
            with self.assertRaises(ValueError):
                d.live()
        with patch.dict("os.environ", {"DYNAMO_URL": ""}, clear=True):
            with self.assertRaises(ValueError):
                d.live()

    def test_live_endpoint_both_url_forms_and_auth(self):
        answer = json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode()
        for suffix, endpoint, key in (
            ("https://example.org/", "https://example.org/v1/chat/completions", ""),
            ("https://example.org/v1/chat/completions", "https://example.org/v1/chat/completions", "secret")
        ):
            seen = []
            def fake_urlopen(request, timeout):
                self.assertEqual(timeout, 45)
                seen.append(request)
                return io.BytesIO(answer)
            with self.subTest(suffix=suffix), patch.dict("os.environ",
                    {"DYNAMO_URL": suffix, "DYNAMO_MODEL": "model-A", "DYNAMO_API_KEY": key}, clear=True), \
                    patch.object(d.urllib.request, "urlopen", side_effect=fake_urlopen), \
                    redirect_stdout(io.StringIO()):
                d.live()
            self.assertEqual(seen[0].full_url, endpoint)
            self.assertEqual(json.loads(seen[0].data)["model"], "model-A")
            self.assertEqual(seen[0].get_method(), "POST")
            if key:
                self.assertIn("Authorization", dict(seen[0].headers))

    def test_live_default_model_and_invalid_response(self):
        with patch.dict("os.environ", {"DYNAMO_URL": "http://example.org"}, clear=True), \
             patch.object(d.urllib.request, "urlopen", return_value=io.BytesIO(b'{"choices":[]}')):
            with self.assertRaises(AssertionError):
                d.live()

    def test_main_contract_and_mocked_live(self):
        for mode in ("--contract-only", "--live"):
            with self.subTest(mode=mode), patch.object(sys,"argv",["dynamo",mode]), \
                 patch.object(d,"live") as live, redirect_stdout(io.StringIO()):
                d.main()
                self.assertEqual(live.called, mode=="--live")
        with patch.object(sys,"argv",["dynamo"]):
            with self.assertRaises(SystemExit):
                d.main()

    def test_script_entrypoint(self):
        buf=io.StringIO()
        with patch.object(sys,"argv",["dynamo","--contract-only"]), redirect_stdout(buf):
            runpy.run_path(str(Path(d.__file__)),run_name="__main__")
        self.assertIn("PASS:",buf.getvalue())


if __name__=="__main__":
    unittest.main()
