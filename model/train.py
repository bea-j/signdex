"""Train the SignSeek classifier on Kaggle (GPU), then export model.onnx, labels.json, similar.json.

Data flow
  prepare.FeatureExtractor.stage1 runs once per sequence (trim, mirror, normalize; NaN kept).
  Training samples:   augment(stage1) -> stage2           (augmentation lives ONLY here)
  Validation/export:  stage2(stage1) == compute(raw)     (exactly what the browser computes)

Split: by participant_id, so validation signers are never seen in training.
Runs: the shipped model is trained WITH augmentation. If the time budget allows, a second model with the
same schedule is trained WITHOUT augmentation, only to report the difference.

Outputs (OUT dir): model.onnx, labels.json, similar.json, metrics.json, parity_fixture.json
"""

from __future__ import annotations

import importlib.util
import json
import math
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
SMOKE = os.environ.get("SIGNSEEK_SMOKE") == "1"
BUDGET_MIN = float(os.environ.get("SIGNSEEK_BUDGET_MIN", "55"))  # whole kernel, wall clock
FIXED_EPOCHS = int(os.environ.get("SIGNSEEK_EPOCHS", "0"))  # >0: train exactly this many epochs (no time planning)
BASELINE = os.environ.get("SIGNSEEK_BASELINE", "1") == "1"  # also train the no-augmentation comparison model
ARCH = os.environ.get("SIGNSEEK_ARCH", "v4")  # "v2": the 65.5% model; "v4": conv+transformer w/ in-model velocity
AUG_NAME = os.environ.get("SIGNSEEK_AUG", "strong")  # "v2" or "strong"
RESERVE_MIN = 5.0  # export, ORT check, similar signs, fixture
MAX_EPOCHS, MIN_EPOCHS = 100, 20
BATCH, WARMUP_EPOCHS = 256, 3
DIM, HEADS, DROP = 256, 4, 0.1
VAL_PARTICIPANTS, SEED = 4, 0

# Per-architecture training settings. v2 reproduces the current model exactly.
ARCH_CFG = {
    "v2": {"lr": 1e-3, "wd": 0.05, "label_smoothing": 0.1, "ema": 0.0, "late_dropout": 0.3, "late_dropout_at": 0.0,
           "depth": 2, "kernel": 5, "drop_path": 0.0},
    # v4, after the top Kaggle ISLR solutions: deeper conv+transformer, wide depthwise kernels, stochastic depth,
    # weight EMA (evaluated and exported), "late dropout" on the head, more label smoothing.
    "v4": {"lr": 1e-3, "wd": 0.05, "label_smoothing": 0.2, "ema": 0.999, "late_dropout": 0.4, "late_dropout_at": 0.4,
           "depth": 3, "kernel": 17, "drop_path": 0.1},
}
CFG = ARCH_CFG[ARCH]

# Augmentation presets, applied per training sample to normalized (shoulder-centered) sequences.
# Keys missing from a preset are off. "v2" reproduces the current model's augmentation exactly.
AUG_PRESETS = {
    "v2": {
        "rotate_deg": 10.0,          # rotation about the shoulder center (the origin after stage1)
        "scale": [0.9, 1.1],         # uniform scale about the shoulder center
        "stretch": [0.85, 1.15],     # independent x and y stretch (camera aspect uncertainty)
        "translate": 0.05,           # +/- shift, in shoulder widths
        "p_trim": 0.5, "trim_frac": 0.1, "trim_max": 4,   # drop up to min(4, 10% of T) frames at each end
        "p_speed": 0.5, "speed": [0.8, 1.2],              # resample to T / speed frames before fixing length
        "p_jitter": 0.5, "jitter_sigma": 0.005,           # Gaussian noise, shoulder-width units
        "p_hand_dropout": 0.3, "dropout_max_frac": 0.3,   # blank one hand for up to 30% of frames
    },
    "strong": {
        "rotate_deg": 15.0, "scale": [0.8, 1.2], "stretch": [0.8, 1.25], "translate": 0.1,
        "shear": 0.15,                                    # x += shear * y and vice versa
        "p_trim": 0.6, "trim_frac": 0.15, "trim_max": 6,
        "p_speed": 0.8, "speed": [0.7, 1.4],
        "p_time_warp": 0.3, "time_warp": 0.25,            # non-uniform speed: move the midpoint by up to 25%
        "p_frame_drop": 0.3, "frame_drop_frac": 0.2,      # drop random frames (lower / uneven capture fps)
        "p_jitter": 0.6, "jitter_sigma": 0.008,
        "p_hand_dropout": 0.4, "dropout_max_frac": 0.4,
        "p_face_dropout": 0.15, "face_dropout_max_frac": 0.5,  # face detection misses (lips + upper face)
    },
}
AUG = AUG_PRESETS[AUG_NAME]


def log(*a) -> None:
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def minutes_left() -> float:
    return BUDGET_MIN - (time.time() - T0) / 60


def ensure_packages() -> None:
    need = [pkg for pkg, mod in [("onnx", "onnx"), ("onnxruntime", "onnxruntime"), ("onnxscript", "onnxscript")]
            if importlib.util.find_spec(mod) is None]
    if need:
        log("pip install", need)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False)


# --------------------------------------------------------------------------- augmentation (train only)

def augment(x: np.ndarray, rng: np.random.Generator, fx: prepare.FeatureExtractor) -> np.ndarray:
    """x: (T, n, 2) normalized, NaN = missing. Returns a new array; NaN stays NaN."""
    T = len(x)
    if T > 2 and rng.random() < AUG["p_trim"]:
        k = min(AUG["trim_max"], int(AUG["trim_frac"] * T))
        if k > 0:
            a, b = (int(v) for v in rng.integers(0, k + 1, size=2))
            if T - a - b >= 1:
                x = x[a: T - b]
                T = len(x)
    if rng.random() < AUG["p_speed"]:
        T2 = max(1, int(round(T / rng.uniform(*AUG["speed"]))))
        x = x[np.minimum(T - 1, ((2 * np.arange(T2) + 1) * T) // (2 * T2))]
        T = T2
    # (new ops below draw random numbers only when enabled, so the "v2" preset reproduces v2 exactly)
    if T > 3 and AUG.get("p_time_warp", 0) > 0 and rng.random() < AUG["p_time_warp"]:
        # piecewise-linear warp: output midpoint maps to an input point moved by up to +/- time_warp
        mid = 0.5 + rng.uniform(-AUG["time_warp"], AUG["time_warp"])
        u = (np.arange(T) + 0.5) / T
        src = np.where(u < 0.5, u / 0.5 * mid, mid + (u - 0.5) / 0.5 * (1 - mid))
        x = x[np.minimum(T - 1, (src * T).astype(np.int64))]
    if T > 3 and AUG.get("p_frame_drop", 0) > 0 and rng.random() < AUG["p_frame_drop"]:
        keep = rng.random(T) >= rng.uniform(0, AUG["frame_drop_frac"])
        if keep.sum() >= 2:
            x = x[keep]
            T = len(x)

    th = np.deg2rad(rng.uniform(-AUG["rotate_deg"], AUG["rotate_deg"]))
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    stretch = np.diag(rng.uniform(*AUG["stretch"], size=2))
    M = stretch @ (rng.uniform(*AUG["scale"]) * rot)
    if AUG.get("shear", 0) > 0:
        sh = rng.uniform(-AUG["shear"], AUG["shear"], size=2)
        M = np.array([[1.0, sh[0]], [sh[1], 1.0]]) @ M
    M = M.astype(np.float32)
    t = rng.uniform(-AUG["translate"], AUG["translate"], size=2).astype(np.float32)
    x = x @ M.T + t

    if rng.random() < AUG["p_jitter"]:
        x = x + rng.normal(0.0, AUG["jitter_sigma"], size=x.shape).astype(np.float32)
    if rng.random() < AUG["p_hand_dropout"]:
        # NaN here becomes 0 with mask 0 in stage2: exactly what a missed detection looks like.
        hand = fx.slices["left_hand" if rng.random() < 0.5 else "right_hand"]
        span = int(rng.integers(1, max(1, int(AUG["dropout_max_frac"] * T)) + 1))
        start = int(rng.integers(0, T - span + 1))
        x[start: start + span, hand] = np.nan
    if AUG.get("p_face_dropout", 0) > 0 and rng.random() < AUG["p_face_dropout"]:
        span = int(rng.integers(1, max(1, int(AUG["face_dropout_max_frac"] * T)) + 1))
        start = int(rng.integers(0, T - span + 1))
        for g in fx.spec["groups"]:
            if g["type"] == "face":  # lips + upper_face: one face detection
                x[start: start + span, fx.slices[g["name"]]] = np.nan
    return x.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- torch parts
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_RNG: np.random.Generator | None = None


def _worker_init(_: int) -> None:
    global _RNG
    _RNG = np.random.default_rng(torch.initial_seed() % 2**32)


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, frames, offsets, idx, y, fx, aug: bool):
        self.frames, self.offsets, self.idx, self.y, self.fx, self.aug = frames, offsets, idx, y, fx, aug

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        global _RNG
        j = self.idx[i]
        x = self.frames[self.offsets[j]: self.offsets[j + 1]]
        if self.aug:
            if _RNG is None:
                _RNG = np.random.default_rng(SEED)
            x = augment(x, _RNG, self.fx)
        return torch.from_numpy(self.fx.stage2(x)), int(self.y[j])


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch per sample while training. Identity in eval/export."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p).to(x.dtype)
        return x * keep / (1.0 - self.p)


class ConvBlock(nn.Module):
    def __init__(self, d: int, k: int = 5, drop: float = DROP, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.expand = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(2 * d, 2 * d, k, padding=k // 2, groups=2 * d)
        self.bn = nn.BatchNorm1d(2 * d)
        self.act = nn.SiLU()
        self.proj = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(drop)
        self.dp = DropPath(drop_path)

    def forward(self, x):  # (B, T, d)
        h = self.expand(self.norm(x)).transpose(1, 2)
        h = self.act(self.bn(self.dw(h))).transpose(1, 2)
        return x + self.dp(self.drop(self.proj(h)))


class AttnBlock(nn.Module):
    """Plain-op self-attention (exports cleanly to ONNX for onnxruntime-web)."""

    def __init__(self, d: int, heads: int = HEADS, drop: float = DROP, drop_path: float = 0.0):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.n1, self.qkv, self.out = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)
        self.dp = DropPath(drop_path)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        a = torch.softmax((q @ k.transpose(-2, -1)) * (self.hd ** -0.5), dim=-1)
        x = x + self.dp(self.drop(self.out((a @ v).transpose(1, 2).reshape(B, T, D))))
        return x + self.dp(self.drop(self.ff(self.n2(x))))


class Net(nn.Module):
    def __init__(self, F: int, L: int, n_classes: int, d: int = DIM):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(F, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        blocks: list[nn.Module] = []
        for _ in range(2):
            blocks += [ConvBlock(d), ConvBlock(d), ConvBlock(d), AttnBlock(d)]
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_classes))

    def forward(self, x):
        h = self.blocks(self.inp(x) + self.pos)
        return self.head(self.norm(h).mean(dim=1))


def coord_mask_matrix(fx: prepare.FeatureExtractor) -> torch.Tensor:
    """(n_masks, 2 * n_points) 0/1: which presence mask covers each x/y coordinate.

    Groups without a mask of their own (upper_face) use the mask of their detector (the face -> lips mask).
    """
    masks = fx.mask_order
    M = torch.zeros(len(masks), 2 * fx.n)
    for g in fx.spec["groups"]:
        name = g["name"] if g["name"] in masks else next(
            h["name"] for h in fx.spec["groups"] if h["type"] == g["type"] and h["name"] in masks)
        s = fx.slices[g["name"]]
        M[masks.index(name), 2 * s.start: 2 * s.stop] = 1.0
    assert (M.sum(0) == 1).all(), "every coordinate must be covered by exactly one mask"
    return M


class NetV4(nn.Module):
    """Conv + transformer after the top Kaggle ISLR solutions, with velocity computed inside the model.

    Input is exactly the per-frame features of feature_spec.json: [points (2N) | masks (M)] per frame.
    The model adds v1 = x[t+1] - x[t] and v2 = x[t+2] - x[t], zeroed wherever either frame lacks the point,
    so the browser's features.ts does not change.
    """

    def __init__(self, F: int, L: int, n_classes: int, cmask: torch.Tensor, d: int = DIM, depth: int = 3,
                 kernel: int = 17, drop_path: float = 0.1):
        super().__init__()
        self.register_buffer("cmask", cmask)
        self.P = cmask.shape[1]
        assert F == self.P + cmask.shape[0]
        self.inp = nn.Sequential(nn.Linear(3 * self.P + cmask.shape[0], d, bias=False), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        n_blocks = 4 * depth
        rates = [drop_path * i / max(1, n_blocks - 1) for i in range(n_blocks)]  # linearly increasing
        blocks: list[nn.Module] = []
        for s in range(depth):
            r = rates[4 * s: 4 * s + 4]
            blocks += [ConvBlock(d, kernel, drop_path=r[0]), ConvBlock(d, kernel, drop_path=r[1]),
                       ConvBlock(d, kernel, drop_path=r[2]), AttnBlock(d, drop_path=r[3])]
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.0), nn.Linear(d, n_classes))  # dropout raised late in training

    def forward(self, x):
        pts, masks = x[..., : self.P], x[..., self.P:]
        m = masks @ self.cmask  # (B, L, 2N): 1 where the coordinate's group was detected
        z1 = torch.zeros_like(pts[:, :1])
        z2 = torch.zeros_like(pts[:, :2])
        v1 = torch.cat([(pts[:, 1:] - pts[:, :-1]) * (m[:, 1:] * m[:, :-1]), z1], dim=1)
        v2 = torch.cat([(pts[:, 2:] - pts[:, :-2]) * (m[:, 2:] * m[:, :-2]), z2], dim=1)
        h = self.blocks(self.inp(torch.cat([pts, v1, v2, masks], dim=-1)) + self.pos)
        return self.head(self.norm(h).mean(dim=1))


def build_model(F: int, L: int, n_classes: int, fx: prepare.FeatureExtractor) -> nn.Module:
    if ARCH == "v2":
        return Net(F, L, n_classes)
    return NetV4(F, L, n_classes, coord_mask_matrix(fx), depth=CFG["depth"], kernel=CFG["kernel"],
                 drop_path=CFG["drop_path"])


class EMA:
    """Exponential moving average of weights; BatchNorm buffers are copied. The EMA copy is evaluated/exported."""

    def __init__(self, model: nn.Module, decay: float):
        import copy

        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float | None = None) -> None:
        d = self.decay if decay is None else decay
        ema_p = list(self.model.parameters())
        torch._foreach_mul_(ema_p, d)
        torch._foreach_add_(ema_p, [p.detach() for p in model.parameters()], alpha=1 - d)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)


class WithSoftmax(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=-1)


def make_loader(frames, offsets, idx, y, fx, aug: bool, shuffle: bool = True):
    workers = min(8, os.cpu_count() or 1)
    return torch.utils.data.DataLoader(
        TrainSet(frames, offsets, idx, y, fx, aug), batch_size=BATCH, shuffle=shuffle, drop_last=True,
        num_workers=workers, persistent_workers=workers > 0, worker_init_fn=_worker_init, pin_memory=True)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), 1024):
        xb = torch.from_numpy(X[i: i + 1024]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(torch.softmax(model(xb).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def topk(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    return float((np.argsort(-probs, axis=1)[:, :k] == y[:, None]).any(axis=1).mean())


def time_steps(loader, model, device, n: int = 40) -> float:
    """Seconds per optimizer step including data loading (throwaway model)."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    model.train()
    it = iter(loader)
    for s in range(n + 5):
        if s == 5:
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.time()
        xb, yb = next(it)
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            loss = nn.functional.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / n


def train_run(name, loader, F, L, n_classes, Xv, yv, epochs, device, fx):
    """Returns (model to evaluate/export, history, minutes). With CFG['ema'] > 0 that is the EMA model."""
    torch.manual_seed(SEED)
    model = build_model(F, L, n_classes, fx).to(device)
    steps = len(loader)
    total, warm = epochs * steps, max(1, WARMUP_EPOCHS * steps)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    ema = EMA(model, CFG["ema"]) if CFG["ema"] > 0 else None
    late_from = int(CFG["late_dropout_at"] * epochs) + 1
    history = []
    t_start = time.time()
    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        model.head[0].p = CFG["late_dropout"] if ep >= late_from else 0.0  # "late dropout" on the head
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if ema is not None:
                # ramp the decay up early on so the average isn't dominated by random init
                ema.update(model, min(CFG["ema"], (1 + step) / (10 + step)))
            tot, n = tot + loss.item() * len(yb), n + len(yb)
        if ep % 10 == 0 or ep == epochs or ep == 1:
            p = predict(ema.model if ema else model, Xv, device)
            row = {"epoch": ep, "train_loss": round(tot / max(n, 1), 4), "val_top1": round(topk(p, yv, 1), 4),
                   "val_top5": round(topk(p, yv, 5), 4)}
            history.append(row)
            eta = (time.time() - t_start) / ep * (epochs - ep) / 60
            log(f"{name} epoch {ep}/{epochs}", row, f"(~{eta:.0f} min left)")
    return (ema.model if ema else model), history, (time.time() - t_start) / 60


def export_onnx(model: nn.Module, path: Path, L: int, F: int) -> str:
    w = WithSoftmax(model.float().cpu().eval()).eval()
    dummy = torch.zeros(1, L, F)
    kw = dict(input_names=["features"], output_names=["probs"],
              dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(w, dummy, str(path), dynamo=False, **kw)
        return "torchscript"
    except Exception as e:  # legacy exporter missing or failing: try the dynamo exporter
        log("legacy ONNX export failed, trying dynamo:", repr(e)[:300])
        torch.onnx.export(w, dummy, str(path), dynamo=True, external_data=False, **kw)
        return "dynamo"


def similar_signs(probs: np.ndarray, y: np.ndarray, labels: list[str], k: int = 3) -> dict:
    """Soft confusion: C[i, j] = mean P(j | true i) over validation; symmetrized, diagonal removed."""
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


def parity_fixture(root, df, val_idx, fx, spec) -> dict:
    """Two real validation sequences (one mirrored) with the features compute() gives, for the JS test."""
    cases, want = [], {True: None, False: None}
    for j in val_idx[np.argsort(np.random.default_rng(SEED).random(len(val_idx)))]:
        raw = np.round(prepare.load_raw(os.path.join(root, df.path.iloc[j])), 6).astype(np.float32)
        if not 10 <= len(raw) <= 30:
            continue
        x = raw[:, fx.gidx, :2]
        pres = fx.presence(x)
        mirrored = bool(pres[:, fx._col("left_hand")].sum() > pres[:, fx._col("right_hand")].sum())
        if want[mirrored] is not None:
            continue
        feats = fx.compute(raw)
        want[mirrored] = {
            "sequence_id": int(df.sequence_id.iloc[j]), "sign": df.sign.iloc[j], "mirrored": mirrored,
            "frames": len(raw),
            "raw_xy": [[[None if np.isnan(v) else float(v) for v in pt] for pt in fr] for fr in raw],
            "features": np.round(feats, 7).tolist(),
        }
        if all(v is not None for v in want.values()):
            break
    cases = [v for v in want.values() if v is not None]
    return {"spec_version": spec["version"], "tolerance": 1e-4,
            "note": "raw_xy is the Kaggle 543-point layout (x, y only; z unused, treat as NaN); null = NaN. "
                    "features = FeatureExtractor.compute(raw), shape [L, F].", "cases": cases}


def check_face_mirror_pairs(root: str, df: pd.DataFrame, spec: dict, n: int = 200, limit: float = 0.08) -> dict:
    """On real frames, each face mirror pair's midpoint must sit on the face midline.

    Offsets are in units of the outer-eye distance (33-263). A correct pair sits within a few percent; a wrong
    pair (both points on one side) sits ~0.2-0.5 away. Raises SystemExit before any training if a pair fails.
    """
    midline = [10, 152, 0, 17, 13, 14]  # forehead, chin, lip centers
    face_groups = {g["name"] for g in spec["groups"] if g["type"] == "face"}
    pairs = [(name, a, b) for name, prs in spec["mirroring"]["swap_pairs"].items() if name in face_groups
             for a, b in prs]
    per_pair: dict[str, list[float]] = {f"{n_}:{a}-{b}": [] for n_, a, b in pairs}
    for path in df.sample(n, random_state=1).path:
        raw = prepare.load_raw(os.path.join(root, path))
        f = raw[~np.isnan(raw[:, 33, 0]) & ~np.isnan(raw[:, 263, 0])]
        if len(f) == 0:
            continue
        width = np.linalg.norm(f[:, 33] - f[:, 263], axis=1)
        mid = f[:, midline, 0].mean(axis=1)
        for name, a, b in pairs:
            per_pair[f"{name}:{a}-{b}"].append(float(np.median(np.abs((f[:, a, 0] + f[:, b, 0]) / 2 - mid) / width)))
    summary = {k: round(float(np.median(v)), 4) for k, v in per_pair.items() if v}
    bad = {k: v for k, v in summary.items() if v > limit}
    log(f"face mirror pairs: worst {max(summary.values()):.4f} (limit {limit}); failing: {bad or 'none'}")
    if bad:
        raise SystemExit(f"face mirror pairs not symmetric on real data: {bad}")
    return {"limit": limit, "worst": max(summary.values()), "per_pair_median_midpoint_offset": summary}


# --------------------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_packages()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", "| torch", torch.__version__,
        "| smoke" if SMOKE else "")

    spec = prepare.load_spec()
    fx = prepare.FeatureExtractor(spec)
    root = prepare.find_data_root()
    df = pd.read_csv(os.path.join(root, "train.csv"))
    with open(os.path.join(root, "sign_to_prediction_index_map.json")) as fh:
        sign_map = json.load(fh)
    labels = [None] * len(sign_map)
    for s, i in sign_map.items():
        labels[i] = s
    log(f"spec v{spec['version']}: {fx.n} points, model input {fx.L} x {fx.F}")
    pair_check = check_face_mirror_pairs(root, df, spec)

    if SMOKE:
        run(df.sample(3000, random_state=SEED).reset_index(drop=True), OUT, "smoke", root, spec, fx, sign_map,
            labels, device, pair_check)
        return
    # End-to-end preflight on a small subset, so a bug in export/ORT/fixture code fails in minutes,
    # not after the full preprocessing + training.
    run(df.sample(400, random_state=SEED).reset_index(drop=True), Path("/tmp/signseek_preflight"), "preflight",
        root, spec, fx, sign_map, labels, device, pair_check)
    log("preflight passed")
    run(df, OUT, "full", root, spec, fx, sign_map, labels, device, pair_check)


def run(df, out: Path, mode: str, root, spec, fx, sign_map, labels, device, pair_check) -> None:
    out.mkdir(parents=True, exist_ok=True)
    y_all = df.sign.map(sign_map).to_numpy(np.int64)

    parts = np.array(sorted(df.participant_id.unique()))
    val_parts = np.sort(np.random.RandomState(SEED).choice(parts, size=VAL_PARTICIPANTS, replace=False))
    is_val = df.participant_id.isin(val_parts).to_numpy()
    tr_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    log(f"{len(df)} sequences, {len(parts)} participants; val participants {val_parts.tolist()}: "
        f"{len(val_idx)} val / {len(tr_idx)} train")

    frames, offsets = prepare.build_stage1([os.path.join(root, p) for p in df.path], spec)
    log("stage1 done:", frames.shape, f"{frames.nbytes / 1e9:.2f} GB")

    Xv = np.stack([fx.stage2(frames[offsets[j]: offsets[j + 1]]) for j in val_idx])  # == compute(raw)
    yv = y_all[val_idx]
    L, F, n_classes = fx.L, fx.F, len(labels)

    aug_loader = make_loader(frames, offsets, tr_idx, y_all, fx, aug=True)
    if mode == "preflight":
        epochs, do_baseline = 1, False
    elif mode == "smoke":
        epochs, do_baseline = 2, True
    elif FIXED_EPOCHS > 0:
        epochs, do_baseline = FIXED_EPOCHS, BASELINE
        log(f"fixed schedule: {epochs} epochs, baseline without augmentation: {do_baseline}")
    else:
        sps = time_steps(aug_loader, build_model(F, L, n_classes, fx).to(device), device)
        epoch_min = sps * len(aug_loader) / 60 * 1.05  # + periodic eval
        avail = minutes_left() - RESERVE_MIN
        both = int(avail / (2 * epoch_min))
        if both >= MIN_EPOCHS:
            epochs, do_baseline = min(MAX_EPOCHS, both), True
        else:
            epochs, do_baseline = max(MIN_EPOCHS, min(MAX_EPOCHS, int(avail / epoch_min))), False
        log(f"{sps * 1000:.0f} ms/step, {epoch_min:.2f} min/epoch, {avail:.0f} min available -> "
            f"{epochs} epochs, baseline without augmentation: {do_baseline}")

    model, hist_aug, min_aug = train_run("aug", aug_loader, F, L, n_classes, Xv, yv, epochs, device, fx)
    probs = predict(model, Xv, device)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- export and verify the shipped (augmented) model
    onnx_path = out / "model.onnx"
    exporter = export_onnx(model, onnx_path, L, F)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    p_onnx = np.concatenate([sess.run(None, {"features": Xv[i: i + 1024]})[0] for i in range(0, len(Xv), 1024)])
    with torch.no_grad():
        p_torch = WithSoftmax(model).eval()(torch.from_numpy(Xv[:512])).numpy()
    onnx_check = {"exporter": exporter, "max_abs_diff_vs_torch_fp32": float(np.abs(p_onnx[:512] - p_torch).max()),
                  "argmax_agreement": float((p_onnx[:512].argmax(1) == p_torch.argmax(1)).mean()),
                  "size_mb": round(onnx_path.stat().st_size / 1e6, 2)}
    log("onnx:", onnx_check)

    per_class = {labels[i]: round(float((p_onnx[yv == i].argmax(1) == i).mean()), 3) for i in range(n_classes)
                 if (yv == i).any()}
    ranked = sorted(per_class.items(), key=lambda kv: kv[1])
    metrics = {
        "mode": mode, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "fixed_epochs": FIXED_EPOCHS or None, "seed": SEED,
        "spec_version": spec["version"], "num_points": fx.n, "input_shape": [L, F],
        "face_mirror_pair_check": pair_check, "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)),
        "val_participants": val_parts.tolist(), "params": int(n_params), "epochs": epochs,
        "with_augmentation": {
            "val_top1_onnx": round(topk(p_onnx, yv, 1), 4), "val_top5_onnx": round(topk(p_onnx, yv, 5), 4),
            "val_top1_torch_fp16": round(topk(probs, yv, 1), 4), "val_top5_torch_fp16": round(topk(probs, yv, 5), 4),
            "train_minutes": round(min_aug, 1), "history": hist_aug},
        "arch": ARCH, "arch_cfg": CFG, "aug_preset": AUG_NAME, "augmentation": AUG, "onnx": onnx_check,
        "worst_10_signs": ranked[:10], "best_10_signs": ranked[-10:][::-1],
    }

    with open(out / "labels.json", "w") as fh:
        json.dump(labels, fh, indent=1)
    with open(out / "similar.json", "w") as fh:
        json.dump(similar_signs(p_onnx, yv, labels), fh, indent=1)
    with open(out / "parity_fixture.json", "w") as fh:
        json.dump(parity_fixture(root, df, val_idx, fx, spec), fh)
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)
    log(f"[{mode}] shipped model:", metrics["with_augmentation"] | {"history": "..."})

    # ---- baseline without augmentation, same schedule, only for the comparison
    need = min_aug * 0.9
    if do_baseline and (mode == "smoke" or minutes_left() - 1 > need):
        del aug_loader
        plain_loader = make_loader(frames, offsets, tr_idx, y_all, fx, aug=False)
        base, hist_base, min_base = train_run("no-aug", plain_loader, F, L, n_classes, Xv, yv, epochs, device, fx)
        pb = predict(base, Xv, device)
        metrics["without_augmentation"] = {"val_top1_torch_fp16": round(topk(pb, yv, 1), 4),
                                           "val_top5_torch_fp16": round(topk(pb, yv, 5), 4),
                                           "train_minutes": round(min_base, 1), "history": hist_base}
    else:
        metrics["without_augmentation"] = {"skipped": f"not enough time ({minutes_left():.0f} min left, "
                                                      f"needed ~{need:.0f})"}
    metrics["total_minutes"] = round((time.time() - T0) / 60, 1)
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)
    log("done:", json.dumps({k: metrics[k] for k in ("with_augmentation", "without_augmentation")},
                            default=str)[:1500])


if __name__ == "__main__":
    main()
