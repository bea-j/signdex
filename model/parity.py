"""Python half of the features parity test (web/test/parity.test.ts drives it).

    python model/parity.py IN.json OUT.json

IN:  {"cases": [{"name": str, "frames": [T][543][2 or 3] with null = NaN}]}
OUT: {"cases": [{"name": str, "features": [L][F]}]}   via prepare.FeatureExtractor.compute

Needs only numpy.
"""

import json
import sys

import numpy as np

from prepare import POINTS_PER_FRAME, FeatureExtractor, load_spec


def to_array(frames: list) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros((0, POINTS_PER_FRAME, 3), dtype=np.float32)
    return np.array([[[np.nan if v is None else v for v in pt] for pt in fr] for fr in frames], dtype=np.float32)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    fx = FeatureExtractor(load_spec())
    out = [{"name": c["name"], "features": fx.compute(to_array(c["frames"])).astype(float).tolist()} for c in cases]
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"cases": out}, fh)


if __name__ == "__main__":
    main()
