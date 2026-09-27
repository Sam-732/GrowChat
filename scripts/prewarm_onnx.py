"""Download and smoke-test the query-time ONNX MiniLM weights.

Run this in the deploy build step, not at request time. `ONNXMiniLM_L6_V2` fetches
~80MB into `~/.cache/chroma/onnx_models/` on first use; doing that during the build
keeps it off the first user's request and fails the build loudly if the download or
the runtime is broken, rather than failing the first question silently.

    python scripts/prewarm_onnx.py

Exits non-zero if the model cannot be loaded or does not return a unit-norm
384-dim vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embed import EMBED_DIM, encode_query, get_onnx_model  # noqa: E402

CACHE_HINT = Path.home() / ".cache" / "chroma" / "onnx_models"


def main() -> int:
    print(f"cache dir: {CACHE_HINT}")
    try:
        get_onnx_model()
    except Exception as exc:
        print(f"FAIL: could not load the ONNX MiniLM model: {exc}", file=sys.stderr)
        return 1

    vector = encode_query("expense ratio")
    norm = sum(value * value for value in vector) ** 0.5
    print(f"vector dim: {len(vector)} (expected {EMBED_DIM})")
    print(f"L2 norm   : {norm:.6f} (expected 1.0)")

    if len(vector) != EMBED_DIM:
        print(f"FAIL: expected {EMBED_DIM} dims, got {len(vector)}", file=sys.stderr)
        return 1
    if abs(norm - 1.0) > 1e-3:
        print(f"FAIL: vector is not unit-norm ({norm})", file=sys.stderr)
        return 1

    print("ok: ONNX MiniLM cached and producing index-compatible vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
