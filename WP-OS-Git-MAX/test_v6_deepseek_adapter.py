"""Offline adapter integration contract tests; not a real DeepSeek call."""
import io
import json
import unittest
from unittest.mock import patch
from v6_deepseek_adapter import analyze


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class DeepSeekAdapterTests(unittest.TestCase):
    def test_mocked_inference_e2e(self):
        def transport(req, timeout):
            self.assertEqual(timeout, 30)
            self.assertEqual(req.get_header("Authorization"), "Bearer test-secret")
            body = json.loads(req.data)
            self.assertEqual(body["model"], "test-model")
            self.assertIn("candidate", body["messages"][1]["content"])
            return FakeResponse(b'{"choices":[{"message":{"content":"reviewed"}}]}')
        self.assertEqual(analyze("candidate scores", endpoint="https://example.org/v1/chat/completions",
            model="test-model", api_key="test-secret", opener=transport), "reviewed")

    def test_missing_credentials_fails_closed(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                analyze("hello")

    def test_http_rejected(self):
        with self.assertRaises(ValueError):
            analyze("hello", endpoint="http://example.org/api", model="m", api_key="secret")


if __name__ == "__main__":
    unittest.main()
