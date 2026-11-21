"""Landmarks -> model features, implemented exactly as model/feature_spec.json describes.

web/src/features.ts must reproduce `FeatureExtractor.compute` within 1e-4 (parity test).

The pipeline is split in two stages so training can augment in between:
  stage1(raw)  select points -> trim hand-less ends -> mirror if left-dominant -> normalize   (NaN kept)
  stage2(x)    resample to L frames (nearest) -> presence masks -> NaN to fill value        -> (L, F)
compute(raw) == stage2(stage1(raw)) is the spec; validation and the browser use exactly that.

CLI (on Kaggle): python prepare.py --out /tmp/signseek_cache   (writes frames.npy, offsets.npy, meta.json)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SPEC_PATH = Path(os.environ.get("SIGNSEEK_SPEC", Path(__file__).with_name("feature_spec.json")))
POINTS_PER_FRAME = 543


def load_spec(path: str | Path = SPEC_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_data_root() -> str:
    for c in ["/kaggle/input/asl-signs", "/kaggle/input/competitions/asl-signs"]:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not hits:
        raise FileNotFoundError("asl-signs train.csv not found under /kaggle/input")
    return os.path.dirname(hits[0])


class FeatureExtractor:
    def __init__(self, spec: dict):
        self.spec = spec
        groups = spec["groups"]
        self.L = int(spec["resample"]["length"])
        self.fill = float(spec["missing"]["fill_value"])
        self.gidx = np.array([i for g in groups for i in g["global_index"]], dtype=np.int64)
        self.n = len(self.gidx)
        assert self.n == spec["num_points"]

        self.slices: dict[str, slice] = {}
        self.local: dict[str, dict[int, int]] = {}  # group -> landmark_index -> position in the n selected points
        self.type_group: dict[str, str] = {}
        o = 0
        for g in groups:
            k = len(g["global_index"])
            self.slices[g["name"]] = slice(o, o + k)
            self.local[g["name"]] = {li: o + j for j, li in enumerate(g["landmark_index"])}
            self.type_group.setdefault(g["type"], g["name"])
            o += k

        mir = spec["mirroring"]
        perm = np.arange(self.n)
        for a, b in mir["swap_groups"]:
            sa, sb = self.slices[a], self.slices[b]
            assert sa.stop - sa.start == sb.stop - sb.start
            perm[sa] = np.arange(sb.start, sb.stop)
            perm[sb] = np.arange(sa.start, sa.stop)
        for gname, pairs in mir["swap_pairs"].items():
            if gname.startswith("_"):
                continue
            for a, b in pairs:
                pa, pb = self.local[gname][a], self.local[gname][b]
                perm[pa], perm[pb] = pb, pa
        assert sorted(perm.tolist()) == list(range(self.n))
        self.mirror_perm = perm

        norm = spec["normalization"]
        cg = self.type_group[norm["center"]["type"]]
        sg = self.type_group[norm["scale"]["type"]]
        self.center_pos = [self.local[cg][i] for i in norm["center"]["landmark_index"]]
        self.scale_pos = [self.local[sg][i] for i in norm["scale"]["landmark_index"]]
        fb = norm["fallback_if_no_shoulders"]
        self.fallback_center = np.array(fb["center"], dtype=np.float64)
        self.fallback_scale = float(fb["scale"])
        self.eps = float(norm["eps"])

        self.mask_order = spec["presence"]["mask_order"]
        self.F = self.n * len(spec["coords"]) + len(self.mask_order)
        assert [self.L, self.F] == spec["output"]["shape"]

    # ---- helpers ----
    def presence(self, x: np.ndarray) -> np.ndarray:
        """(T, n, 2) -> (T, len(mask_order)) bool: group present if any of its x values is not NaN."""
        return np.stack([~np.isnan(x[:, self.slices[g], 0]).all(axis=1) for g in self.mask_order], axis=1)

    def _col(self, name: str) -> int:
        return self.mask_order.index(name)

    # ---- spec stages ----
    def stage1(self, raw: np.ndarray) -> np.ndarray:
        """raw (T, 543, >=2) Kaggle layout -> (T', n, 2) float32, trimmed, mirrored, normalized, NaN kept."""
        raw = np.asarray(raw)
        assert raw.ndim == 3 and raw.shape[1] == POINTS_PER_FRAME, raw.shape
        x = raw[:, self.gidx, :2].astype(np.float32, copy=True)
        if len(x) == 0:
            return x

        pres = self.presence(x)
        lh, rh = self._col("left_hand"), self._col("right_hand")
        hand = pres[:, lh] | pres[:, rh]
        if hand.any():
            i = np.flatnonzero(hand)
            x, pres = x[i[0]: i[-1] + 1], pres[i[0]: i[-1] + 1]

        if pres[:, lh].sum() > pres[:, rh].sum():
            x = x[:, self.mirror_perm]
            x[..., 0] = 1.0 - x[..., 0]

        a, b = x[:, self.center_pos[0]].astype(np.float64), x[:, self.center_pos[1]].astype(np.float64)
        mid = (a + b) / 2.0
        sa, sb = x[:, self.scale_pos[0]].astype(np.float64), x[:, self.scale_pos[1]].astype(np.float64)
        dist = np.sqrt(((sa - sb) ** 2).sum(axis=1))
        valid = np.isfinite(mid).all(axis=1) & np.isfinite(dist)
        if valid.any():
            c, s = mid[valid].mean(axis=0), dist[valid].mean()
        else:
            c, s = self.fallback_center, self.fallback_scale
        s = max(float(s), self.eps)
        x = ((x - c.astype(np.float32)) / np.float32(s)).astype(np.float32)
        return x

    def resample_index(self, T: int) -> np.ndarray:
        i = np.arange(self.L, dtype=np.int64)
        return np.minimum(T - 1, ((2 * i + 1) * T) // (2 * self.L))

    def stage2(self, x: np.ndarray) -> np.ndarray:
        """(T, n, 2) NaN-kept -> (L, F) float32 model input."""
        if len(x) == 0:
            return np.zeros((self.L, self.F), dtype=np.float32)
        x = x[self.resample_index(len(x))]
        masks = self.presence(x).astype(np.float32)
        pts = np.nan_to_num(x, nan=self.fill).reshape(self.L, -1)
        return np.concatenate([pts, masks], axis=1).astype(np.float32)

    def compute(self, raw: np.ndarray) -> np.ndarray:
        return self.stage2(self.stage1(raw))


def load_raw(path: str, coords=("x", "y")) -> np.ndarray:
    """Kaggle parquet -> (T, 543, len(coords)). Row order verified by kaggle/datacheck."""
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=list(coords))
    arr = np.stack([t.column(c).to_numpy().astype(np.float32) for c in coords], axis=-1)
    assert len(arr) % POINTS_PER_FRAME == 0, (path, len(arr))
    return arr.reshape(-1, POINTS_PER_FRAME, len(coords))


_FX: FeatureExtractor | None = None


def _init_worker(spec: dict) -> None:
    global _FX
    _FX = FeatureExtractor(spec)


def _stage1_file(path: str) -> np.ndarray:
    assert _FX is not None
    return _FX.stage1(load_raw(path))


def build_stage1(paths: list[str], spec: dict, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run stage1 over many parquet files. Returns frames (sumT, n, 2) and offsets (N+1,)."""
    workers = workers or os.cpu_count() or 1
    with Pool(workers, initializer=_init_worker, initargs=(spec,)) as pool:
        seqs = list(pool.imap(_stage1_file, paths, chunksize=64))
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    frames = np.concatenate(seqs, axis=0) if len(seqs) else np.zeros((0, spec["num_points"], 2), np.float32)
    return frames, offsets


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="/tmp/signseek_cache")
    args = ap.parse_args()
    root = args.root or find_data_root()
    spec = load_spec()
    df = pd.read_csv(os.path.join(root, "train.csv"))
    frames, offsets = build_stage1([os.path.join(root, p) for p in df.path], spec)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "frames.npy"), frames)
    np.save(os.path.join(args.out, "offsets.npy"), offsets)
    print("frames", frames.shape, "sequences", len(offsets) - 1)


if __name__ == "__main__":
    main()
