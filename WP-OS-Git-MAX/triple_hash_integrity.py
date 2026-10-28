"""Canonical triple digest for research prediction payloads (never changes ticket selection)."""
import hashlib
import json

def canonical_bytes(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")

def triple_hash(payload):
    try:
        import blake3
    except ImportError as exc:
        raise RuntimeError("BLAKE3 required: pip install blake3") from exc
    raw = canonical_bytes(payload)
    return {"sha256": hashlib.sha256(raw).hexdigest(),
            "sha3_256": hashlib.sha3_256(raw).hexdigest(),
            "blake3": blake3.blake3(raw).hexdigest()}

def verify_hash(payload, digests):
    actual = triple_hash(payload)
    return (set(actual) == set(digests) and
            all(__import__('hmac').compare_digest(actual[k], digests[k]) for k in actual))
