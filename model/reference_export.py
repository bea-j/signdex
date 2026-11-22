"""Reference animations for all 250 signs, picked from the full training data (Kaggle CPU kernel).

    python kaggle/bundle.py --entry reference && kaggle kernels push -p kaggle/reference

For every training-signer sequence (validation participants excluded, as in train.py): features stage 1 (as in
features.ts), hand visibility per frame, and the v4 model's probabilities. Per sign, keyed by its label:
  - the sign counts as two-handed when at least 10% of its recordings show both hands in 30% or more of their frames
    (MediaPipe often misses the non-dominant hand, so a typical recording under-reports it);
  - visibility = fraction of frames with both hands (two-handed signs) or with the dominant hand (one-handed);
  - among sequences the model classifies correctly with >= 12 frames, take the best visibility, then the highest
    confidence (fallbacks: >= 8 frames, then the best probability for the sign).
Writes signs.bin (model/reference.py format) and selection.json to /kaggle/working.
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
import reference

T0 = time.time()
OUT = Path(os.environ.get("SIGNSEEK_OUT", "/kaggle/working"))
SEED, VAL_PARTICIPANTS = 0, 4
MIN_FRAMES, MIN_FRAMES_FALLBACK = 12, 8
TWO_HANDED_FRAMES, TWO_HANDED_SHARE = 0.3, 0.9  # two-handed: >= 10% of recordings show both hands in >= 30% of frames


def log(*a) -> None:
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("onnxruntime") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=False)
    import onnxruntime as ort

    spec = prepare.load_spec()
    root = prepare.find_data_root()
    df = pd.read_csv(os.path.join(root, "train.csv"))
    with open(os.path.join(root, "sign_to_prediction_index_map.json")) as fh:
        sign_map = json.load(fh)
    labels = [None] * len(sign_map)
    for s, i in sign_map.items():
        labels[i] = s

    parts = np.array(sorted(df.participant_id.unique()))
    val = set(np.random.RandomState(SEED).choice(parts, size=VAL_PARTICIPANTS, replace=False).tolist())
    for mf in glob.glob("/kaggle/input/**/metrics_v4.json", recursive=True):  # same split as the model's training
        rec = json.load(open(mf)).get("val_participants")
        assert rec is None or sorted(rec) == sorted(val), (mf, rec, sorted(val))
    tdf = df[~df.participant_id.isin(val)].reset_index(drop=True)
    log(f"training signers only ({len(tdf)} sequences; validation participants {sorted(val)} excluded)")

    model_path = sorted(glob.glob("/kaggle/input/**/v4.onnx", recursive=True))[0]
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    L = sess.get_inputs()[0].shape[1]
    fx = prepare.FeatureExtractor(spec, length=L)
    log("model", model_path, "frames", L)

    frames, offsets = prepare.build_stage1([os.path.join(root, p) for p in tdf.path], spec)
    log("stage 1 done:", frames.shape)
    seqs = [frames[offsets[j]: offsets[j + 1]] for j in range(len(tdf))]

    # hand visibility per sequence (stage-1 point order starts with left_hand 21, right_hand 21)
    T = np.array([len(s) for s in seqs])
    lh = np.array([(~np.isnan(s[:, 0:21, 0]).all(1)).mean() if len(s) else 0.0 for s in seqs])
    rh = np.array([(~np.isnan(s[:, 21:42, 0]).all(1)).mean() if len(s) else 0.0 for s in seqs])
    both = np.array([((~np.isnan(s[:, 0:21, 0]).all(1)) & (~np.isnan(s[:, 21:42, 0]).all(1))).mean() if len(s) else 0.0
                     for s in seqs])
    dominant = np.maximum(lh, rh)  # after stage 1 the dominant hand is the right slot, but take the max to be safe

    probs = np.zeros((len(seqs), len(labels)), dtype=np.float32)
    for i in range(0, len(seqs), 512):
        X = np.stack([fx.stage2(s) for s in seqs[i: i + 512]])
        probs[i: i + len(X)] = sess.run(None, {"features": X})[0]
    log("model done")
    y = tdf.sign.map(sign_map).to_numpy()
    pred = probs.argmax(1)
    p_true = probs[np.arange(len(y)), y]

    sign_arr = tdf.sign.to_numpy()
    chosen, stats = {}, {}
    for sign in labels:
        idx = np.flatnonzero(tdf.sign.to_numpy() == sign)
        if len(idx) == 0:
            log(f"  {sign}: no training sequences, skipped")
            stats["missing"] = stats.get("missing", 0) + 1
            continue
        two = float(np.quantile(both[idx], TWO_HANDED_SHARE)) >= TWO_HANDED_FRAMES
        vis = both if two else dominant
        how = None
        for min_t, need_correct, name in ((MIN_FRAMES, True, "correct"), (MIN_FRAMES_FALLBACK, True, "correct_short"),
                                          (MIN_FRAMES_FALLBACK, False, "best_probability")):
            c = idx[(T[idx] >= min_t) & ((pred[idx] == sign_map[sign]) if need_correct else True)]
            if len(c):
                if need_correct:  # best visibility (5% steps), then the dominant hand (5% steps), then confidence
                    best = c[np.lexsort((-p_true[c], -np.round(dominant[c] * 20), -np.round(vis[c] * 20)))[0]]
                else:
                    best = c[np.argmax(p_true[c])]
                how = name
                break
        if how is None:
            best, how = idx[np.argmax(p_true[idx])], "any"
        stats[how] = stats.get(how, 0) + 1
        chosen[sign] = {
            "s1": seqs[best], "p": float(p_true[best]), "selection": how, "two_handed": two,
            "visibility": float(vis[best]), "both_frac": float(both[best]), "hand_frac": float(dominant[best]),
            "sequence_id": int(tdf.sequence_id[best]), "participant_id": int(tdf.participant_id[best]),
            "frames": int(T[best]), "candidates": int(len(idx)),
            "correct_candidates": int((pred[idx] == sign_map[sign]).sum()),
        }
    out = reference.pack(chosen, OUT / "signs.bin", spec, L, Path(model_path).name)

    # shortlist for re-selecting locally without another kernel run: per sign the top 15 correct candidates by
    # both-hands and by dominant-hand visibility, and the 5 most confident; stage-1 frames as float16 + stats
    keep = set()
    sign_arr = tdf.sign.to_numpy()
    for sign in labels:
        idx = np.flatnonzero(sign_arr == sign)
        ok = idx[(T[idx] >= MIN_FRAMES_FALLBACK) & (pred[idx] == sign_map[sign])]
        keep.update(ok[np.argsort(-both[ok])][:15].tolist())
        keep.update(ok[np.argsort(-dominant[ok])][:15].tolist())
        keep.update(idx[np.argsort(-p_true[idx])][:5].tolist())
    keep = sorted(keep)
    sl_off = np.cumsum([0] + [len(seqs[i]) for i in keep])
    np.savez_compressed(
        OUT / "shortlist.npz",
        frames=np.concatenate([seqs[i] for i in keep]).astype(np.float16), offsets=sl_off,
        sign=np.array([sign_arr[i] for i in keep]), sequence_id=tdf.sequence_id.to_numpy()[keep],
        participant_id=tdf.participant_id.to_numpy()[keep], T=T[keep], left=lh[keep], right=rh[keep], both=both[keep],
        p=p_true[keep], correct=(pred[keep] == y[keep]),
        both_p90=np.array([float(np.quantile(both[sign_arr == sign_arr[i]], TWO_HANDED_SHARE)) for i in keep]))
    log(f"shortlist: {len(keep)} sequences, {(OUT / 'shortlist.npz').stat().st_size / 1e6:.0f} MB")
    with open(OUT / "selection.json", "w") as fh:
        json.dump({s: {k: v for k, v in c.items() if k != "s1"} for s, c in chosen.items()}, fh, indent=1)
    vis_all = np.array([c["visibility"] for c in chosen.values()])
    log(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB): {len(chosen)} signs, selection {stats}, "
        f"two-handed signs {sum(c['two_handed'] for c in chosen.values())}, visibility median {np.median(vis_all):.2f}, "
        f"min {vis_all.min():.2f}")
    log("two-handed signs:", " ".join(sorted(s for s, c in chosen.items() if c["two_handed"])))
    p90 = {sg: float(np.quantile(both[sign_arr == sg], TWO_HANDED_SHARE)) for sg in labels if (sign_arr == sg).any()}
    log("both-hands p90 (two-handed if >= %.2f): " % TWO_HANDED_FRAMES + ", ".join(
        f"{sg} {p90[sg]:.2f}" for sg in ("book", "shoe", "bath", "clean", "finish", "happy", "cat", "drink", "yes", "hat")
        if sg in p90))
    for s in ("book", "hat", "cat", "drink", "yes"):
        if s not in chosen:
            continue
        c = chosen[s]
        log(f"  {s}: {c['selection']}, p={c['p']:.3f}, two-handed={c['two_handed']}, visibility={c['visibility']:.2f}, "
            f"both hands {c['both_frac']:.2f}, {c['frames']} frames, from {c['correct_candidates']}/{c['candidates']} correct")


if __name__ == "__main__":
    main()
