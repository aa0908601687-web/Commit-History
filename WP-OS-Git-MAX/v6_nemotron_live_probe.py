#!/usr/bin/env python3
"""Fail-closed live NVIDIA Nemotron inference verification; no prediction changes."""
import json
import os
import sys
import urllib.error
import urllib.request

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3-super-120b-a12b"

def validate_response(body, requested_model=MODEL):
    if not isinstance(body, dict):
        raise ValueError("response must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("response did not identify a model")
    if model != requested_model:
        raise ValueError("model mismatch: expected configured Nemotron model")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing model completion")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("empty or invalid model completion")
    if "NEMOTRON_OK" not in answer:
        raise ValueError("unexpected live inference result")
    return {"status": "live_nemotron_inference_verified",
            "model": model, "nonempty_completion": True, "live_inference": True}

def probe(key, opener=None, model=MODEL):
    if not key or not key.strip():
        raise ValueError("NVIDIA_API_KEY secret is not configured; live E2E not verified")
    payload = json.dumps({"model": model,
        "messages": [{"role": "user", "content": "Respond with exactly NEMOTRON_OK"}],
        "max_tokens": 80, "temperature": 0, "stream": False}).encode("utf-8")
    req = urllib.request.Request(URL, data=payload, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = opener or urllib.request.urlopen
    with opener(req, timeout=90) as response:
        if response.status != 200:
            raise ValueError("NVIDIA endpoint returned non-200 HTTP response")
        body = json.loads(response.read(1048576).decode("utf-8"))
    return validate_response(body, model)

def main():
    try:
        result = probe(os.environ.get("NVIDIA_API_KEY", ""))
        print(json.dumps(result, sort_keys=True))
    except (ValueError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        # Never display server responses or exception strings: they might expose credentials.
        print(json.dumps({"status": "live_nemotron_inference_unverified",
                          "error_type": type(exc).__name__, "live_inference": False}))
        sys.exit(1)

if __name__ == "__main__":
    main()
