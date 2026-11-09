"""V6 Dynamo inference contract and optional live endpoint smoke test.

Dynamo is a serving layer, not a lottery scoring model. No QAOA.
"""
import argparse
import json
import os
import urllib.request
from urllib.parse import urlparse


def payload(model="test-model"):
    return {"model": model, "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 16, "temperature": 0}


def validate(response):
    assert isinstance(response, dict)
    choices = response.get("choices")
    assert isinstance(choices, list) and choices
    assert isinstance(choices[0].get("message", {}).get("content"), str)
    return True


def live():
    url = os.environ.get("DYNAMO_URL", "")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("DYNAMO_URL must be a valid http(s) endpoint")
    model = os.environ.get("DYNAMO_MODEL") or "default"
    headers = {"Content-Type": "application/json"}
    if os.environ.get("DYNAMO_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["DYNAMO_API_KEY"]
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/v1/chat/completions"
    req = urllib.request.Request(endpoint, data=json.dumps(payload(model)).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=45) as response:
        result = json.load(response)
    validate(result)
    print("PASS: live inference endpoint returned a valid chat completion")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract-only", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.contract_only:
        assert payload()["messages"][0]["role"] == "user"
        validate({"choices": [{"message": {"content": "OK"}}]})
        print("PASS: inference request/response contract only; no Dynamo server contacted")
    else:
        live()


if __name__ == "__main__":
    main()
