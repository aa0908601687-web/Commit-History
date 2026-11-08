import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("v6_nemotron_vision_client.py")
spec = importlib.util.spec_from_file_location("nemotron_client", MODULE)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class VisionClientTests(unittest.TestCase):
    def test_parse_json_fence(self):
        self.assertEqual(client.parse_completion('```json\n{"draws":[]}\n```'), {"draws":[]})

    def test_mocked_image_api(self):
        with tempfile.TemporaryDirectory() as d:
            image=Path(d)/"draw.png"
            image.write_bytes(b"fake-png-for-mocked-request")
            class Response(io.BytesIO):
                def __enter__(self): return self
                def __exit__(self, *args): self.close()
            def opener(request, timeout=90):
                assert request.get_header("Authorization")=="Bearer fake-key"
                payload=json.loads(request.data)
                assert payload["model"]==client.MODEL
                assert payload["messages"][0]["content"][1]["type"]=="image_url"
                response={"choices":[{"message":{"content":json.dumps({"draws":[{"draw_id":"115000001","first":[1,2,3,4,5,6],"second":1}]})}}]}
                return Response(json.dumps(response).encode())
            result=client.extract(image,"fake-key",opener=opener)
            self.assertEqual(result["draws"][0]["second"],1)

    def test_no_key_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"draw.png"
            p.write_bytes(b"fake")
            with self.assertRaisesRegex(ValueError,"NVIDIA_API_KEY"):
                client.extract(p,"")


if __name__=="__main__":
    unittest.main()
