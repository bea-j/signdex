"""Turn Kaggle asl-signs parquet files into fixed-size feature tensors.

Reads feature_spec.json (the same file web/src/features.ts uses) and writes X (N, 32, 182), y (N,),
signer ids for a held-out-signer split, and labels.json.

Not implemented yet.
"""

import json
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("feature_spec.json")


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit("prepare.py: not implemented yet")
