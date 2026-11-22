# Signdex status

_Updated Sat Nov 21, 8:45 PM ET. Hacking ends Sun 12:00 PM._

Live: **https://signdex.study** (main app) and https://signdex.study/debug.html (landmark debug page).

## Model in use: v4 (69.0% top-1, 83.7% top-5)

All numbers are on the same validation set: 4 held-out participants (4718, 28656, 36257, 62590), 17,521
sequences never seen in training. Scored independently by the `signseek-evaluate` Kaggle kernel (version 2),
which reproduced the training runs' own numbers.

| Model | Frames | top-1 | top-5 | top-1 / top-5 with 5-version averaging | Size |
|---|---|---|---|---|---|
| v2 (previous) | 32 | 65.5% | 82.3% | 65.7% / 82.5% | 11 MB |
| **v4 (in use)** | **64** | **69.0%** | **83.7%** | 68.9% / 83.4% | 17 MB |
| v5 (same setup as v4) | 64 | 68.9% | 83.7% | 68.7% / 83.5% | 17 MB |
| v4 + v5 ensemble | 64 | 69.8% | 84.6% | 69.6% / 84.3% | 34 MB |
| Target (CLAUDE.md) | | >70% | >90% | | |

- **v4** is Kaggle kernel `beatricej/signseek-train` version 4: conv + transformer, velocity inside the model,
  stronger augmentation, 245 epochs in 2.3 h on T4 ×2. Input `features [batch, 64, 222]`, output
  `probs [batch, 250]`; the exported ONNX matches torch exactly.
- **Ensemble:** +0.8 top-1 over v4 for twice the size and lookup time; kept v4 (rule: switch at +1 point).
- **Test-time averaging does not help v4** (68.9% vs 69.0% single pass) and costs 5× the model time
  (~197 ms vs ~41 ms per lookup in onnxruntime-web). It did help v2 slightly. The app still averages 5 versions;
  switching to 1 is a one-line change (`TTA_VARIANTS` in `web/src/app/webcam.ts` and `trim.ts`).
- Hardest signs for v4: snack 10%, goose 18%, puzzle 20%, there 20%, awake 21%, kiss 23%, sleepy 26%.
- Files: `model/model.onnx`, `labels.json`, `similar.json`, `metrics.json`, `parity_fixture.json` (spec v4),
  `training.log`. **Not in git.** The previous v2 files are backed up in `model/runs/v2-backup/`; every run's
  output is in `model/runs/kernel-v4/`, `kernel-v5/`, and the evaluation in `model/runs/eval-v2/`.

## Web app (signdex.study)

- **Home:** the Figma Make design (Manrope self-hosted, design colors as CSS variables in `web/src/app.css`),
  with a looping demo in the hero panel: a reference skeleton signs, its hands light up, then the result
  card fills in. Still frame with `prefers-reduced-motion`.
- **Webcam and uploaded clip:** one screen each, the camera/player large on the left and the top 5 in a panel
  to the right (below on phones), updated in place. Webcam: camera picker, Space/R → 3-2-1 → 2 s recording →
  lookup; the panel replays the recorded clip. Upload: drag a 1–3 s window on a thumbnail strip (mouse,
  touch, arrow keys), Space/R looks it up.
- **Switches (remembered):** "Show hand & body points" and "Show face points" draw the debug page's overlay on
  the live camera, the replay and the uploaded clip; "Show details" shows camera/landmark/lookup stats.
- **Result cards:** display name and meaning from `dictionary.json`, confidence colors (high ≥ 60%, medium
  ≥ 40%, low), "Looks similar to" from `similar.json`, "Watch a real signer" link, Save (My signs list on the
  device), and a **reference animation**: a PopSign training signer's landmarks as a mirrored skeleton (top
  card loops, others play on hover/tap). Under 40%: "Not sure…".
- **Reference animations:** `model/reference.py` (candidates → download → select with v4) writes
  `web/public/reference/signs.bin` (not in git; data). **Only 66 of 250 signs so far**: Kaggle rate-limits
  single-file competition downloads (404s after a few hundred files). Cards for the other signs show no
  animation; the home demo tops up missing demo signs (hat, drink, yes) with other confident ones. To finish:
  `model/.venv/Scripts/python model/reference.py download` when Kaggle allows it again, then `… select`.
- **Dictionary:** edit `demo/dictionary.csv`, then `npm run dictionary -w web`; it refuses to write unless the
  labels exactly match `labels.json`.

## Chrome extension

`npm run build -w extension`, load `extension/dist` unpacked. Alt+Shift+S on a video tab starts capture (page
`<video>` when readable, otherwise tab capture cropped to the video); **press it again** (or "Look up last N s",
slider 1–5 s) to look up. Same model, cards, dictionary and reference animations as the web app.

## Deploy

```sh
bash deploy/deploy.sh                          # asks before each server command
bash deploy/deploy.sh --yes build upload verify   # redeploy a new build
```

Nginx + Let's Encrypt on the Vultr server (`SIGNDEX_HOST`, default in the script). Releases are kept in
`/var/www/signdex/releases/` (newest 3) with `current` pointing to the live one.

## Tests

`npm test -w web`: 30 tests (`features.ts` vs `prepare.py` at 64 and 32 frames, the Kaggle fixture, raw
landmarks → `features.ts` → onnxruntime-web matching Kaggle-feature predictions, test-time averaging). Python
side: `model/.venv`.

## Not yet verified

- A real webcam lookup in the new layout, and the extension's second-hotkey lookup, have only been tested
  with recorded landmark sequences (headless Chrome's fake camera has no person in it; Chrome no longer loads
  unpacked extensions from the command line).

## Next steps

1. **Try it for real** in Chrome on signdex.study (webcam lookups, an uploaded clip) and the extension on YouTube.
2. **Reference animations for all 250 signs** (see above).
3. **Decide on test-time averaging** for v4 (off = same accuracy, 5× faster).
4. **Final model:** retrain v4 on all 21 participants for the demo build (no held-out set, so no new number).
5. **Demo:** 3–5 pre-tested clips in `demo/` with expected results.
