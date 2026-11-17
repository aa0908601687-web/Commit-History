"""Optional DeepSeek-compatible OpenAI-style HTTP inference adapter.

No model is installed by this module. An actual endpoint and credentials are
required for live inference. Never place API keys in source control.
"""
import json
import os
import urllib.request
from urllib.parse import urlparse


def analyze(prompt, *, endpoint=None, model=None, api_key=None, opener=None):
    endpoint = endpoint or os.getenv("V6_DEEPSEEK_ENDPOINT")
    model = model or os.getenv("V6_DEEPSEEK_MODEL")
    api_key = api_key or os.getenv("V6_DEEPSEEK_API_KEY")
    if not endpoint or not model or not api_key:
        raise RuntimeError("DeepSeek endpoint, model and API key are required")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("DeepSeek endpoint must be a valid HTTPS URL without embedded credentials")
    payload = json.dumps({"model": model, "messages": [
        {"role": "system", "content": "Review candidate scores and code logic; do not claim lottery prediction advantage."},
        {"role": "user", "content": prompt}], "stream": False}).encode()
    req = urllib.request.Request(endpoint, data=payload,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST")
    transport = opener or urllib.request.urlopen
    with transport(req, timeout=30) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"]
