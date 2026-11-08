#!/usr/bin/env python3
"""Optional NVIDIA Nemotron vision API -> checked V6 draw JSON.

Requires an explicitly configured NVIDIA_API_KEY and a supported NVIDIA VLM.
Never writes credentials to logs. Does not modify canonical draw history.
"""
import argparse
import base64
import json
import os
import re
import urllib.request
from pathlib import Path

MODEL = "nvidia/nemotron-nano-12b-v2-vl"
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
PROMPT = (
    "Read lottery results from the image. Return ONLY a JSON object of the form "
    '{"draws":[{"draw_id":"NNNNNNNNN","first":[1,2,3,4,5,6],"second":1}]}. '
    "Use only clearly visible draw IDs and six first-zone numbers, plus one second-zone "
    "number. Do not guess or fabricate uncertain values. Use an empty draws list when "
    "the image does not contain a readable complete draw."
)


def parse_completion(message):
    if isinstance(message, list):
        message = "".join(x.get("text", "") for x in message if isinstance(x, dict))
    if not isinstance(message, str):
        raise ValueError("non-text VLM completion")
    cleaned = message.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, dict) or not isinstance(data.get("draws"), list):
        raise ValueError("Nemotron response must contain draws array")
    return data


def extract(image_path, api_key, model=MODEL, opener=None):
    path = Path(image_path)
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        raise ValueError("Only JPG PNG WEBP images supported")
    raw = path.read_bytes()
    if len(raw) > 8_000_000 or not raw:
        raise ValueError("image size must be 1..8000000 bytes")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is required")
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png" if suffix == ".png" else "image/webp"
    uri = "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": uri}}]}],
        "temperature": 0, "max_tokens": 2048, "stream": False}
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST")
    response = (opener or urllib.request.urlopen)(req, timeout=90)
    with response as handle:
        answer = json.load(handle)
    try:
        text = answer["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected NVIDIA API response") from exc
    extracted = parse_completion(text)
    if not extracted["draws"]:
        raise ValueError("No fully readable draw; refusing to generate an empty input")
    return {"source": "nvidia-nemotron-vision:" + model, "draws": extracted["draws"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    result = extract(args.image, os.environ.get("NVIDIA_API_KEY", ""), args.model)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote extracted JSON; cross-check against trusted CSV before running V6.")


if __name__ == "__main__":
    main()
