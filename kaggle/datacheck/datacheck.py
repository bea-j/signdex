"""SignSeek data check (Kaggle script kernel).

Answers the open questions in model/feature_spec.md on ~500 sampled sequences and writes datacheck.json:
  1. hand slots: does Kaggle left_hand sit near pose wrist 15 or 16?
  2. aspect ratio: raw x/y ranges, plus a rotation-invariance estimate of H/W from rigid hand bones
  3. median shoulder width and frame-count distribution (raw and after trimming hand-less frames)
Also verifies the 543-row layout assumption and the lip mirror pairs.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

N_SAMPLE = 500
OUT = "/kaggle/working/datacheck.json"


def find_root() -> str:
    for c in ["/kaggle/input/asl-signs", "/kaggle/input/competitions/asl-signs"]:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not hits:
        raise SystemExit("train.csv not found under /kaggle/input")
    return os.path.dirname(hits[0])


ROOT = find_root()
print("data root:", ROOT)

FACE, LH, POSE, RH = 0, 468, 489, 522
P = lambda i: POSE + i  # noqa: E731
EXPECTED_TYPES = np.array(["face"] * 468 + ["left_hand"] * 21 + ["pose"] * 33 + ["right_hand"] * 21)
EXPECTED_IDX = np.concatenate([np.arange(468), np.arange(21), np.arange(33), np.arange(21)])

LIP_PAIRS = [[61, 291], [185, 409], [40, 270], [39, 269], [37, 267], [146, 375], [91, 321], [181, 405],
             [84, 314], [78, 308], [191, 415], [80, 310], [81, 311], [82, 312], [95, 324], [88, 318],
             [178, 402], [87, 317]]
HAND_BONES = [(0, 5), (0, 17), (5, 17), (0, 9), (5, 9), (9, 13), (13, 17)]  # palm: rigid
ASPECTS = np.round(np.exp(np.linspace(np.log(0.4), np.log(2.5), 61)), 4)  # candidate H/W


def pct(a, qs=(1, 5, 25, 50, 75, 95, 99)):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs} | {"mean": round(float(a.mean()), 4), "n": int(a.size)}


train = pd.read_csv(f"{ROOT}/train.csv")
print("train.csv:", len(train), "sequences,", train.participant_id.nunique(), "participants,", train.sign.nunique(), "signs")
sample = train.sample(N_SAMPLE, random_state=0).reset_index(drop=True)

layout_ok = 0
layout_bad = []
frames_sorted = 0
seq = []  # (T, 543, 3) arrays
for path in sample.path:
    df = pd.read_parquet(f"{ROOT}/{path}", columns=["frame", "type", "landmark_index", "x", "y", "z"])
    T = len(df) // 543
    ok = len(df) == T * 543 and (df.type.values == np.tile(EXPECTED_TYPES, T)).all() and (
        df.landmark_index.values == np.tile(EXPECTED_IDX, T)).all()
    if ok:
        layout_ok += 1
    else:
        layout_bad.append(path)
    fr = df.frame.values.reshape(T, 543)[:, 0] if ok else df.frame.values
    frames_sorted += int(np.all(np.diff(fr) > 0)) if ok else 0
    seq.append(df[["x", "y", "z"]].to_numpy(np.float32).reshape(T, 543, 3) if ok else None)

print(f"layout: {layout_ok}/{N_SAMPLE} match the 543-row face/left_hand/pose/right_hand order; frames strictly increasing in {frames_sorted}")
seq = [s for s in seq if s is not None]

present = lambda s, off, n: ~np.isnan(s[:, off:off + n, 0]).all(axis=1)  # noqa: E731

# ---------- missingness ----------
miss = {"left_hand": [], "right_hand": [], "lips": [], "pose": [], "pose_wrists": [], "shoulders": []}
for s in seq:
    miss["left_hand"].append(present(s, LH, 21).mean())
    miss["right_hand"].append(present(s, RH, 21).mean())
    miss["lips"].append((~np.isnan(s[:, 61, 0])).mean())
    miss["pose"].append(present(s, POSE, 33).mean())
    miss["pose_wrists"].append((~np.isnan(s[:, [P(15), P(16)], 0]).any(axis=1)).mean())
    miss["shoulders"].append((~np.isnan(s[:, [P(11), P(12)], 0]).all(axis=1)).mean())
presence = {k: round(float(np.mean(v)), 4) for k, v in miss.items()}
print("fraction of frames present:", presence)

# ---------- 1. hand slot vs pose wrist ----------
hand_slots = {}
for name, off in [("left_hand", LH), ("right_hand", RH)]:
    d15, d16, hx = [], [], []
    for s in seq:
        m = present(s, off, 21) & ~np.isnan(s[:, P(15), 0]) & ~np.isnan(s[:, P(16), 0])
        if not m.any():
            continue
        w = s[m, off, :2]
        d15.append(np.linalg.norm(w - s[m, P(15), :2], axis=1))
        d16.append(np.linalg.norm(w - s[m, P(16), :2], axis=1))
        mid = (s[m, P(11), 0] + s[m, P(12), 0]) / 2
        hx.append(w[:, 0] - mid)
    d15, d16, hx = map(np.concatenate, (d15, d16, hx))
    hand_slots[name] = {
        "frames": int(d15.size),
        "frac_nearer_pose15": round(float((d15 < d16).mean()), 4),
        "median_dist_pose15": round(float(np.median(d15)), 4),
        "median_dist_pose16": round(float(np.median(d16)), 4),
        "median_x_minus_midshoulder": round(float(np.median(hx)), 4),
    }
dominant = {"left_more": 0, "right_more": 0, "tie": 0}
for s in seq:
    nl, nr = present(s, LH, 21).sum(), present(s, RH, 21).sum()
    dominant["left_more" if nl > nr else "right_more" if nr > nl else "tie"] += 1
pose_side = {
    "median_x_pose15_minus_pose16": round(float(np.nanmedian(np.concatenate([s[:, P(15), 0] - s[:, P(16), 0] for s in seq]))), 4),
    "median_x_pose11_minus_pose12": round(float(np.nanmedian(np.concatenate([s[:, P(11), 0] - s[:, P(12), 0] for s in seq]))), 4),
}
print("hand slots:", json.dumps(hand_slots, indent=1))
print("dominant slot per sequence:", dominant, "| pose sides:", pose_side)

# ---------- 2. coordinate ranges + aspect ratio ----------
ranges = {}
for name, idx in [("all_points", slice(None)), ("hands", np.r_[LH:LH + 21, RH:RH + 21]),
                  ("shoulders", [P(11), P(12)]), ("nose", [P(0)]), ("lips", [l for p in LIP_PAIRS for l in p])]:
    xs = np.concatenate([s[:, idx, 0].ravel() for s in seq])
    ys = np.concatenate([s[:, idx, 1].ravel() for s in seq])
    ranges[name] = {"x": pct(xs, (1, 50, 99)), "y": pct(ys, (1, 50, 99))}
print("ranges:", json.dumps(ranges, indent=1))


def bone_score(aspect_scan, off_list, bones):
    """Median over (sequence, hand, bone) of std(log 3D length) across frames, for each candidate H/W."""
    groups = []
    for s in seq:
        for off in off_list:
            m = present(s, off, 21)
            if m.sum() < 8:
                continue
            h = s[m, off:off + 21, :]
            for a, b in bones:
                groups.append(h[:, b] - h[:, a])  # (n, 3): dx, dy, dz (z in x units)
    scores = []
    for asp in aspect_scan:
        stds = [np.std(np.log(np.sqrt(v[:, 0] ** 2 + (asp * v[:, 1]) ** 2 + v[:, 2] ** 2) + 1e-9)) for v in groups]
        scores.append(float(np.median(stds)))
    return np.array(scores), len(groups)


scores, n_groups = bone_score(ASPECTS, [LH, RH], HAND_BONES)
best = float(ASPECTS[int(np.argmin(scores))])
# same without z, as a sanity check
scores_xy = []
for asp in ASPECTS:
    st = []
    for s in seq:
        for off in [LH, RH]:
            m = present(s, off, 21)
            if m.sum() < 8:
                continue
            h = s[m, off:off + 21, :]
            for a, b in HAND_BONES:
                v = h[:, b] - h[:, a]
                st.append(np.std(np.log(np.sqrt(v[:, 0] ** 2 + (asp * v[:, 1]) ** 2) + 1e-9)))
    scores_xy.append(float(np.median(st)))
best_xy = float(ASPECTS[int(np.argmin(scores_xy))])

# per-sequence estimates (where enough hand frames)
per_seq_best = []
for s in seq:
    for off in [LH, RH]:
        m = present(s, off, 21)
        if m.sum() < 15:
            continue
        h = s[m, off:off + 21, :]
        sc = []
        for asp in ASPECTS:
            sc.append(np.mean([np.std(np.log(np.sqrt((h[:, b, 0] - h[:, a, 0]) ** 2 + (asp * (h[:, b, 1] - h[:, a, 1])) ** 2
                                                     + (h[:, b, 2] - h[:, a, 2]) ** 2) + 1e-9)) for a, b in HAND_BONES]))
        per_seq_best.append(float(ASPECTS[int(np.argmin(sc))]))
named = {"9:16 portrait (H/W 1.778)": 1.778, "3:4 portrait (1.333)": 1.333, "1:1 (1.0)": 1.0,
         "4:3 landscape (0.75)": 0.75, "16:9 landscape (0.5625)": 0.5625}
nearest_named = {k: 0 for k in named}
for b in per_seq_best:
    k = min(named, key=lambda k: abs(np.log(b) - np.log(named[k])))
    nearest_named[k] += 1
aspect = {
    "method": "argmin over candidate H/W of median std(log 3D palm-bone length) across frames",
    "best_H_over_W_xyz": best, "best_H_over_W_xy_only": best_xy, "groups": n_groups,
    "score_at_named": dict(zip(named, (round(float(x), 5) for x in bone_score(np.array(list(named.values())), [LH, RH], HAND_BONES)[0]))),
    "curve": {str(a): round(float(sc), 5) for a, sc in zip(ASPECTS[::4], scores[::4])},
    "per_sequence_best": pct(per_seq_best, (10, 25, 50, 75, 90)),
    "per_sequence_nearest_named": nearest_named,
}
print("aspect:", json.dumps(aspect, indent=1))

# ---------- 3. shoulder width + frame counts ----------
sw_seq, sw_frame = [], []
for s in seq:
    d = np.linalg.norm(s[:, P(11), :2] - s[:, P(12), :2], axis=1)
    sw_frame.append(d)
    if np.isfinite(d).any():
        sw_seq.append(np.nanmean(d))
sw_frame = np.concatenate(sw_frame)
# shoulder width in "x units" after correcting y by the estimated aspect (for reference)
sw_corr = np.concatenate([np.sqrt((s[:, P(11), 0] - s[:, P(12), 0]) ** 2 + (best * (s[:, P(11), 1] - s[:, P(12), 1])) ** 2) for s in seq])
shoulder = {"per_sequence_mean": pct(sw_seq), "per_frame": pct(sw_frame), "per_frame_aspect_corrected": pct(sw_corr)}

T_raw, T_trim, no_hands = [], [], 0
for s in seq:
    T_raw.append(len(s))
    anyhand = present(s, LH, 21) | present(s, RH, 21)
    if not anyhand.any():
        no_hands += 1
        T_trim.append(len(s))
        continue
    i = np.flatnonzero(anyhand)
    T_trim.append(int(i[-1] - i[0] + 1))
T_raw, T_trim = np.array(T_raw), np.array(T_trim)
frames = {
    "raw": pct(T_raw), "trimmed": pct(T_trim), "sequences_without_any_hand": no_hands,
    "trimmed_frac_gt": {str(k): round(float((T_trim > k).mean()), 4) for k in (16, 24, 32, 48, 64, 96)},
}
print("shoulder:", json.dumps(shoulder, indent=1))
print("frames:", json.dumps(frames, indent=1))

# ---------- lip mirror pairs ----------
lip_err_y, lip_err_mid, wrong_err_y = [], [], []
for s in seq:
    m = ~np.isnan(s[:, 61, 0])
    if not m.any():
        continue
    f = s[m]
    width = np.linalg.norm(f[:, 61, :2] - f[:, 291, :2], axis=1)
    midx = (f[:, 0, 0] + f[:, 17, 0] + f[:, 13, 0] + f[:, 14, 0]) / 4
    for a, b in LIP_PAIRS:
        lip_err_y.append(np.median(np.abs(f[:, a, 1] - f[:, b, 1]) / width))
        lip_err_mid.append(np.median(np.abs((f[:, a, 0] + f[:, b, 0]) / 2 - midx) / width))
    for a, b in [[61, 185], [40, 39], [146, 91]]:  # deliberately wrong pairs, for scale
        wrong_err_y.append(np.median(np.abs(f[:, a, 1] - f[:, b, 1]) / width))
lips = {
    "pairs_median_abs_dy_over_mouth_width": round(float(np.median(lip_err_y)), 4),
    "pairs_p95_abs_dy_over_mouth_width": round(float(np.percentile(lip_err_y, 95)), 4),
    "pairs_median_midpoint_offset_over_mouth_width": round(float(np.median(lip_err_mid)), 4),
    "wrong_pairs_median_abs_dy_over_mouth_width": round(float(np.median(wrong_err_y)), 4),
}
print("lips:", lips)

out = {
    "n_sequences_total": int(len(train)), "n_participants": int(train.participant_id.nunique()),
    "n_signs": int(train.sign.nunique()), "n_sampled": N_SAMPLE,
    "layout_ok": layout_ok, "layout_bad": layout_bad[:10], "frames_strictly_increasing": frames_sorted,
    "presence": presence, "hand_slots": hand_slots, "dominant_slot": dominant, "pose_side": pose_side,
    "ranges": ranges, "aspect": aspect, "shoulder_width": shoulder, "frame_counts": frames, "lip_pairs": lips,
}
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", OUT)
