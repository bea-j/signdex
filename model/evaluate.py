"""Evaluate exported ONNX models on the held-out validation signers (Kaggle CPU kernel).

For every *.onnx in the attached dataset (single models and merged ensembles from ensemble.py):
  - top-1 / top-5 on the validation participants, at the frame length L the model declares
  - the same with test-time averaging over 5 trimmed versions, exactly as web/src/lookup.ts does it
  - similar_<name>.json (soft confusion, as in train.py) and the 10 worst signs
Writes eval_metrics.json with all candidates and the best one (by single-pass top-1, then top-5).

The validation split is rebuilt like train.py (seed 0, 4 participants) and checked against the
val_participants recorded in each model's metrics_<name>.json if present.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import prepare

T0 = time.time()
OUT = Path(os.environ.get("SIGNSEEK_OUT", "/kaggle/working"))
SEED, VAL_PARTICIPANTS, TTA = 0, 4, 5


def log(*a) -> None:
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def ensure_ort() -> None:
    if importlib.util.find_spec("onnxruntime") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=False)


def val_split(df: pd.DataFrame) -> np.ndarray:
    """Same as train.py: sorted participants, RandomState(0).choice(4)."""
    parts = np.array(sorted(df.participant_id.unique()))
    return np.sort(np.random.RandomState(SEED).choice(parts, size=VAL_PARTICIPANTS, replace=False))


def tta_offsets(T: int, n: int) -> list[tuple[int, int]]:
    """Mirror of ttaOffsets() in web/src/lookup.ts."""
    out = [(0, 0)]
    if n <= 1 or T < 8:
        return out
    d = max(1, int(np.floor(0.06 * T + 0.5)))  # == Math.max(1, Math.round(0.06 * T)); Python round() is banker's
    out += [(d, 0), (0, d), (2 * d, 0), (0, 2 * d)]
    return out[: min(n, len(out))]


def trim(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """Mirror of trimStage1() in web/src/features.ts."""
    T = len(x)
    if T == 0:
        return x
    a = max(0, min(a, T - 1))
    b = max(0, min(b, T - 1 - a))
    return x[a: T - b]


def topk(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    return float((np.argsort(-probs, axis=1)[:, :k] == y[:, None]).any(axis=1).mean())


def similar_signs(probs: np.ndarray, y: np.ndarray, labels: list[str], k: int = 3) -> dict:
    n = len(labels)
    C = np.zeros((n, n))
    for i in range(n):
        m = y == i
        if m.any():
            C[i] = probs[m].mean(axis=0)
    S = (C + C.T) / 2
    np.fill_diagonal(S, 0)
    return {labels[i]: [{"sign": labels[j], "score": round(float(S[i, j]), 4)} for j in np.argsort(-S[i])[:k]]
            for i in range(n)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_ort()
    import onnxruntime as ort

    spec = prepare.load_spec()
    root = prepare.find_data_root()
    df = pd.read_csv(os.path.join(root, "train.csv"))
    with open(os.path.join(root, "sign_to_prediction_index_map.json")) as fh:
        sign_map = json.load(fh)
    labels = [None] * len(sign_map)
    for s, i in sign_map.items():
        labels[i] = s

    models = sorted(p for p in glob.glob("/kaggle/input/**/*.onnx", recursive=True))
    if not models:
        raise SystemExit("no .onnx files in the attached datasets")
    log("models:", [os.path.relpath(m, "/kaggle/input") for m in models])

    val_parts = val_split(df)
    for mf in glob.glob("/kaggle/input/**/metrics_*.json", recursive=True):
        rec = json.load(open(mf)).get("val_participants")
        assert rec is None or sorted(rec) == val_parts.tolist(), (mf, rec, val_parts.tolist())
    vdf = df[df.participant_id.isin(val_parts)].reset_index(drop=True)
    yv = vdf.sign.map(sign_map).to_numpy(np.int64)
    log(f"validation: participants {val_parts.tolist()}, {len(vdf)} sequences")

    frames, offsets = prepare.build_stage1([os.path.join(root, p) for p in vdf.path], spec)
    seqs = [frames[offsets[j]: offsets[j + 1]] for j in range(len(vdf))]
    log("stage1 done:", frames.shape)

    extractors: dict[int, prepare.FeatureExtractor] = {}
    plain_cache: dict[int, np.ndarray] = {}

    def plain(L: int) -> np.ndarray:  # == FeatureExtractor(spec, L).compute(raw) for every val sequence
        if L not in plain_cache:
            fx = extractors.setdefault(L, prepare.FeatureExtractor(spec, length=L))
            plain_cache[L] = np.stack([fx.stage2(s) for s in seqs])
        return plain_cache[L]

    results: dict[str, dict] = {}
    for path in models:
        name = Path(path).stem
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        L = sess.get_inputs()[0].shape[1]
        fx = extractors.setdefault(L, prepare.FeatureExtractor(spec, length=L))
        run = lambda X: np.concatenate([sess.run(None, {"features": X[i: i + 512]})[0]  # noqa: E731
                                        for i in range(0, len(X), 512)])
        t = time.time()
        probs = run(plain(L))
        single_s = time.time() - t

        # test-time averaging, exactly like lookup.ts: variants after stage 1, probabilities averaged
        t = time.time()
        tta = np.zeros_like(probs)
        for start in range(0, len(seqs), 1024):
            chunk = range(start, min(start + 1024, len(seqs)))
            Xs, owner = [], []
            for i in chunk:
                for a, b in tta_offsets(len(seqs[i]), TTA):
                    Xs.append(fx.stage2(trim(seqs[i], a, b)))
                    owner.append(i)
            p = run(np.stack(Xs))
            owner = np.array(owner)
            np.add.at(tta, owner, p)
            counts = np.bincount(owner - start, minlength=len(chunk))[: len(chunk)]
            tta[start: start + len(chunk)] /= counts[:, None]
        tta_s = time.time() - t

        per_class = {labels[i]: round(float((probs[yv == i].argmax(1) == i).mean()), 3) for i in range(len(labels))
                     if (yv == i).any()}
        results[name] = {
            "file": os.path.relpath(path, "/kaggle/input"), "frames": int(L),
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "top1": round(topk(probs, yv, 1), 4), "top5": round(topk(probs, yv, 5), 4),
            "tta5_top1": round(topk(tta, yv, 1), 4), "tta5_top5": round(topk(tta, yv, 5), 4),
            "worst_10_signs": sorted(per_class.items(), key=lambda kv: kv[1])[:10],
            "cpu_seconds_single_pass": round(single_s, 1), "cpu_seconds_tta5": round(tta_s, 1),
        }
        with open(OUT / f"similar_{name}.json", "w") as fh:
            json.dump(similar_signs(probs, yv, labels), fh, indent=1)
        np.save(OUT / f"val_probs_{name}.npy", probs.astype(np.float16))
        log(name, {k: v for k, v in results[name].items() if k != "worst_10_signs"})

    best = max(results, key=lambda n: (results[n]["top1"], results[n]["top5"]))
    out = {"n_val": int(len(vdf)), "val_participants": val_parts.tolist(), "tta_variants": TTA,
           "models": results, "best_single_pass": best}
    with open(OUT / "eval_metrics.json", "w") as fh:
        json.dump(out, fh, indent=1)
    log("best:", best, results[best]["top1"], results[best]["top5"])


if __name__ == "__main__":
    main()
