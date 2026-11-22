"""Reference animations for the result cards: one representative training sequence per sign.

    python model/reference.py candidates            # train.csv -> K random candidates per sign (training signers only)
    python model/reference.py download              # those parquet files, with the Kaggle API (no kernel)
    python model/reference.py select --model model/model.onnx [--out web/public/reference/signs.bin]

Selection, per sign: candidates whose frames mostly show a hand (>= MIN_HAND_FRAC of frames); among those, the one the
model classifies correctly with the highest confidence (fallback: highest probability for the right sign). Each
chosen sequence goes through the same stage 1 as features.ts (trim hand-less ends, mirror left-dominant signers,
center on mid-shoulder, scale by shoulder width), keeps hands, lips and pose, and is quantized to int8.

File format (little endian): b"SDXR", uint32 header length, header JSON (utf-8), then per sign an int8 block of
shape [frames, points, 2] (x, y; -128 = missing) in units of `scale` shoulder widths. The header lists the signs
(label, offset, frames, confidence, source sequence), the kept points (group and Kaggle landmark index), and
`aspect_y` (multiply y by it to draw with the training video's aspect).
Source data: Google's Isolated Sign Language Recognition (Kaggle asl-signs), PopSign ASL v1.0, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import prepare  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "model" / "runs" / "reference"
RAW = WORK / "raw"
COMPETITION = "asl-signs"
K = 6  # candidates per sign
SEED = 0
VAL_PARTICIPANTS = 4  # same split as train.py / evaluate.py
MIN_HAND_FRAC = 0.6
MIN_FRAMES = 8
MAX_FRAMES = 48
SCALE = 0.025  # shoulder widths per int8 step (range about +-3.2)
KEEP_GROUPS = ("left_hand", "right_hand", "lips", "pose")


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    a = KaggleApi()
    a.authenticate()
    return a


def fetch(a, name: str, dest: Path) -> Path:
    """One competition file -> dest (unzipping if the API delivers a zip)."""
    target = dest / name  # the API keeps the competition path (e.g. train_landmark_files/<pid>/<id>.parquet)
    if target.exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):  # Kaggle throttles bursts of single-file downloads (answers 404 meanwhile)
        try:
            a.competition_download_file(COMPETITION, name, path=str(dest), quiet=True)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * 3**attempt)
    for z in (dest / (name + ".zip"), dest / (Path(name).name + ".zip")):
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target.parent)
            z.unlink()
    if not target.exists():
        raise FileNotFoundError(f"{name}: download produced no {target.name}")
    return target


def val_participants(rows: list[dict]) -> set[int]:
    parts = np.array(sorted({int(r["participant_id"]) for r in rows}))
    return set(np.random.RandomState(SEED).choice(parts, size=VAL_PARTICIPANTS, replace=False).tolist())


def cmd_candidates(_args) -> None:
    a = api()
    train_csv = fetch(a, "train.csv", WORK)
    rows = list(csv.DictReader(open(train_csv, newline="")))
    val = val_participants(rows)
    by_sign: dict[str, list[dict]] = {}
    for r in rows:
        if int(r["participant_id"]) not in val:
            by_sign.setdefault(r["sign"], []).append(r)
    rng = np.random.RandomState(SEED)
    cands = {}
    for sign in sorted(by_sign):
        pool = by_sign[sign]
        pick = rng.choice(len(pool), size=min(K, len(pool)), replace=False)
        cands[sign] = [{k: pool[i][k] for k in ("path", "participant_id", "sequence_id")} for i in sorted(pick)]
    (WORK / "candidates.json").write_text(json.dumps(cands, indent=1))
    print(f"{len(cands)} signs, {sum(map(len, cands.values()))} candidates (training signers only; validation "
          f"participants {sorted(val)} excluded) -> {WORK / 'candidates.json'}")


def cmd_download(args) -> None:
    cands = json.loads((WORK / "candidates.json").read_text())
    todo = [c["path"] for cs in cands.values() for c in cs]
    a = api()
    done = 0

    def one(path: str):
        return fetch(a, path, RAW)

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for path, fut in zip(todo, [ex.submit(one, p) for p in todo]):
            try:
                fut.result()
            except Exception as e:  # keep going; select skips missing files
                errors.append(f"{path}: {e}")
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    size = sum(f.stat().st_size for f in RAW.rglob("*.parquet"))
    print(f"downloaded {len(todo) - len(errors)}/{len(todo)} files, {size / 1e6:.0f} MB in {RAW}")
    for e in errors[:10]:
        print("  error:", e)


def cmd_select(args) -> None:
    import onnxruntime as ort

    spec = prepare.load_spec()
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    L = sess.get_inputs()[0].shape[1]
    fx = prepare.FeatureExtractor(spec, length=L)
    labels = json.loads((REPO / "model" / "labels.json").read_text())
    index = {s: i for i, s in enumerate(labels)}
    cands = json.loads((WORK / "candidates.json").read_text())

    # points kept from the stage-1 layout (spec group order), with their Kaggle landmark index
    keep_pos, points, pos = [], [], 0
    for g in spec["groups"]:
        for gi in g["global_index"]:
            if g["name"] in KEEP_GROUPS:
                keep_pos.append(pos)
                points.append([g["name"], int(gi)])
            pos += 1

    chosen, stats = {}, {"correct": 0, "fallback_wrong": 0, "fallback_hands": 0, "missing": 0}
    for sign in sorted(cands):
        scored = []
        for c in cands[sign]:
            f = RAW / c["path"]
            if not f.exists():
                continue
            try:
                raw = prepare.load_raw(str(f))
            except Exception as e:  # e.g. a file still being downloaded
                print(f"  skip {c['path']}: {e}")
                continue
            hands = ~np.isnan(raw[:, 468:489, 0]).all(1) | ~np.isnan(raw[:, 522:543, 0]).all(1)
            s1 = fx.stage1(raw)
            probs = sess.run(None, {"features": fx.stage2(s1)[None]})[0][0]
            scored.append({**c, "hand_frac": float(hands.mean()), "frames": len(s1), "s1": s1,
                           "p": float(probs[index[sign]]), "top": labels[int(probs.argmax())],
                           "top_p": float(probs.max())})
        if not scored:
            stats["missing"] += 1
            continue
        good = [s for s in scored if s["hand_frac"] >= MIN_HAND_FRAC and s["frames"] >= MIN_FRAMES]
        correct = [s for s in good if s["top"] == sign]
        if correct:
            best, how = max(correct, key=lambda s: s["p"]), "correct"
        elif good:
            best, how = max(good, key=lambda s: s["p"]), "fallback_wrong"
        else:
            best, how = max(scored, key=lambda s: (s["hand_frac"], s["p"])), "fallback_hands"
        stats[how] += 1
        chosen[sign] = {**best, "selection": how}

    # quantize and pack
    blocks, signs, offset = [], [], 0
    for sign in sorted(chosen):
        c = chosen[sign]
        x = c["s1"][:, keep_pos, :]
        if len(x) > MAX_FRAMES:  # same nearest resample formula as features
            i = np.arange(MAX_FRAMES)
            x = x[np.minimum(len(x) - 1, ((2 * i + 1) * len(x)) // (2 * MAX_FRAMES))]
        q = np.where(np.isnan(x), -128, np.clip(np.round(x / SCALE), -127, 127)).astype(np.int8)
        blocks.append(q.tobytes())
        signs.append({"label": sign, "offset": offset, "frames": int(len(q)), "confidence": round(c["p"], 4),
                      "selection": c["selection"], "hand_frac": round(c["hand_frac"], 3),
                      "sequence_id": int(c["sequence_id"]), "participant_id": int(c["participant_id"])})
        offset += len(blocks[-1])
    header = {
        "format": "signdex-reference-v1", "scale": SCALE, "aspect_y": spec["input_aspect"]["train_H_over_W"],
        "fps": 20, "points": points, "model": Path(args.model).name, "model_frames": L,
        "source": "Google Isolated Sign Language Recognition (Kaggle asl-signs), PopSign ASL v1.0, CC BY 4.0",
        "transform": "features stage 1: trimmed, left-dominant mirrored, centered on mid-shoulder, / shoulder width",
        "signs": signs,
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"SDXR" + struct.pack("<I", len(hb)) + hb + b"".join(blocks))
    print(f"wrote {out}: {len(signs)} signs, {out.stat().st_size / 1e3:.0f} kB; selection {stats}")
    for sign in ("hat", "cat", "drink", "yes"):
        if sign in chosen:
            c = chosen[sign]
            print(f"  {sign}: p={c['p']:.3f} ({c['selection']}), {c['frames']} frames, hands in {c['hand_frac']:.0%}")
    (WORK / "selection.json").write_text(json.dumps(
        {s: {k: v for k, v in c.items() if k != "s1"} for s, c in chosen.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates")
    d = sub.add_parser("download")
    d.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("select")
    s.add_argument("--model", default=str(REPO / "model" / "model.onnx"))
    s.add_argument("--out", default=str(REPO / "web" / "public" / "reference" / "signs.bin"))
    args = ap.parse_args()
    os.chdir(REPO)
    {"candidates": cmd_candidates, "download": cmd_download, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()
