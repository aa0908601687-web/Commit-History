#!/usr/bin/env python3
"""Live Hugging Face model registry connectivity check, NOT inference."""
import argparse
import json
import urllib.error
import urllib.request

MODELS = {
    "chronos_2": "amazon/chronos-2",
    "t0_gamma": "theforecastingcompany/t0-gamma",
}


def check_model(model_id, opener=urllib.request.urlopen, timeout=12):
    url = "https://huggingface.co/api/models/" + model_id
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "WP-OS-ABE-connectivity-check/1.0",
                          "Accept": "application/json"})
        with opener(request, timeout=timeout) as response:
            data = json.load(response)
        if data.get("id") != model_id:
            raise ValueError("Unexpected registry model identity")
        return {"registry_reachable": True, "repository": model_id,
                "revision": data.get("sha"), "inference_verified": False}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as err:
        return {"registry_reachable": False, "repository": model_id,
                "error": str(err), "inference_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    models = {name: check_model(identifier) for name, identifier in MODELS.items()}
    report = {
        "kind": "registry_connectivity_only",
        "models": models,
        "lstm": "local model not implemented",
        "gru": "local model not implemented",
        "all_registries_reachable": all(v["registry_reachable"] for v in models.values()),
        "full_five_model_inference_verified": False,
        "note": "Checking a model repository is not downloading, loading or executing it."
    }
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    if not report["all_registries_reachable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
