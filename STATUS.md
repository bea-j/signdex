# Signdex status

_Updated Sat Nov 21, 4:20 PM ET. Hacking ends Sun 12:00 PM._

## Model results so far

The model in use is `model/model.onnx`: the "v2" architecture trained on feature spec v3 (32 frames), Kaggle
kernel `beatricej/signseek-train` **version 2**. Validation uses 4 held-out participants (4718, 28656,
36257, 62590), 17,521 sequences, never seen in training.

| Model | top-1 | top-5 |
|---|---|---|
| **v2, with augmentation (in use)** | **65.5%** | **82.3%** |
| v2, same schedule, no augmentation | 63.4% | 80.0% |
| Target (CLAUDE.md) | >70% | >90% |

- 2.8M parameters, 66 epochs, 20 min on one T4. ONNX is 11.2 MB, opset 17, input `features [batch, 32, 222]`,
  output `probs [batch, 250]`. Checked in onnxruntime (Python) and onnxruntime-web: the exported model gives
  the same predictions as torch.
- Worst signs (top-1): snack 8%, goose 12%, sleepy 16%, there 17%, will 18%, kiss 20%, puzzle 21%,
  beside 23%, mitten 26%, look 30%.
- Most confused pairs (from `similar.json`, soft confusion): awake/wake, duck/goose, stay/that, lips/mouth,
  cut/scissors, pen/pencil, bird/goose, finger/wait, gift/give, sleep/sleepy. Several are the same or nearly
  the same sign in ASL, so part of the gap to the target is the vocabulary itself; these are also exactly the
  "looks similar to" chips we want to show.
- Full numbers and training curves: `model/metrics.json`. Log: `model/training.log`.
- Model files are **not in git** (`model.onnx`, `labels.json`, `similar.json`, `metrics.json`,
  `parity_fixture.json`, `training.log`). Get them from a teammate, or from the version 2 output on the
  Kaggle kernel page.

## v4 Kaggle run (in progress)

Kaggle kernel `beatricej/signseek-train`, **version 4**: GPU **T4 x2** (DataParallel across both GPUs,
mixed precision fp16), **2.5-hour budget** (`-t 10200` hard timeout as a backstop).

- **Model "v4"**, after the top Kaggle ISLR solutions:
  - conv + transformer: 3 stages × (3 depthwise-conv blocks, kernel 17, + 1 attention block), width 256
  - velocity (`x[t+1]−x[t]`, `x[t+2]−x[t]`) computed **inside the model**, masked where a frame is missing
  - stochastic depth, weight EMA (evaluated and exported), late head dropout, label smoothing 0.2
- **Feature spec v4: 64 frames** (was 32). The per-frame features are unchanged, so `features.ts` only takes a
  length parameter.
- **Stronger augmentation:** rotation, scale, x/y stretch, shear, shift, speed 0.7–1.4×, time warp, frame
  drops, hand and face dropouts, jitter.
- **Epochs are planned from the budget** after timing a real training step; the log shows the planned count.
- **Same validation participants as v2.**
- **Status:** queued as of 4:19 PM ET. Versions 1 (old smoke test) and 3 (an earlier 10-hour v4 run) were
  cancelled.
- **Decision rule:** v4 replaces the current model only if it has a higher validation top-1 (top-5 is reported
  too). Back up the current `model/model.onnx` before replacing it.

**How to check it:**

```sh
kaggle kernels status beatricej/signseek-train/4      # QUEUED / RUNNING / COMPLETE / ERROR
kaggle kernels logs -f beatricej/signseek-train/4     # live log while running (Ctrl+C to stop)
# when COMPLETE, download into a NEW folder (not model/) and compare before replacing anything:
kaggle kernels output beatricej/signseek-train -p /tmp/signseek-v4
python -c "import json;m=json.load(open('/tmp/signseek-v4/metrics.json'))['with_augmentation'];print(m['val_top1_onnx'],m['val_top5_onnx'])"
```

In the log, look for `preflight passed` (a 1-epoch end-to-end check on 400 sequences, in the first minutes)
and then `N GPU(s), … min/epoch, … -> E epochs`.

If the Kaggle CLI fails with `CERTIFICATE_VERIFY_FAILED` (an antivirus scanning HTTPS traffic), set
`REQUESTS_CA_BUNDLE` to a CA bundle that includes that antivirus's root certificate.

**How it was pushed** (to reproduce or re-run):

```sh
python kaggle/bundle.py --arch v4 --aug strong --no-baseline --budget 150
kaggle kernels push -p kaggle/train --accelerator NvidiaTeslaT4 -t 10200
```

Kaggle allows only 2 batch GPU sessions at a time, and queued runs count toward that. The CLI cannot cancel a
run; cancel it from the kernel's Versions list on kaggle.com.

## How to run the debug page

```sh
npm install
npm run dev          # first run downloads the MediaPipe models and copies model/ + the ORT runtime into web/public/
```

Open <http://localhost:5173/debug.html>.

1. Click **Use webcam**, or pick a video file. The overlay shows the landmarks the model sees: the
   left-hand slot in orange, the right-hand slot in blue, upper-body pose, and a faint lip outline. Tick
   "Show face points" for the face mesh.
2. Click **Record 2 s and look up** and sign one sign. The top 5 signs appear with confidence; below 40% it
   says it is not sure.
3. **Download last 6 s as JSON** saves a landmark recording. Put it in `demo/landmarks/` and the parity test
   picks it up.

The page reads the frame length from the loaded `model.onnx`, so it works with the current 32-frame model and
a 64-frame v4 model. Without `model/model.onnx` the page still shows landmarks and says no model is loaded.

**Tests:** `npm test -w web` (26 tests: `features.ts` vs `prepare.py` at 64 and 32 frames, Kaggle fixture,
and raw landmarks → `features.ts` → onnxruntime-web matching Kaggle-feature predictions). The Python side uses
`model/.venv` (numpy, onnx, onnxruntime).

**Chrome extension:** `npm run build -w extension`, then load `extension/dist` unpacked and press Alt+Shift+S
on a video. It uses the page `<video>` element when readable, and otherwise tab capture cropped to the video.

## Not yet verified

- The debug page lookup and the extension capture build and pass tests, but **neither has been tried in a real
  browser yet**.

## Next steps

1. **v4 run:** when it finishes, compare its top-1/top-5 with 65.5/82.3. Swap in `model.onnx` only if top-1
   is higher.
2. **Try the debug page and the extension in Chrome** (webcam lookup; YouTube capture path, landmark fps).
3. **Main web app (MUST):** upload/webcam, trim a 1–3 s window, results panel with gloss, confidence bars,
   signasl.org / handspeak.com search links, "looks similar to" chips from `similar.json`, low-confidence
   message.
4. **Extension:** run the model on the captured landmark buffer and show results in the side panel.
5. **Final model:** retrain the winning setup on all 21 participants for the demo build (one more run).
6. **Demo:** 3–5 pre-tested clips in `demo/` with expected results; credit PopSign ASL v1.0 (CC BY 4.0) and the
   Kaggle competition in the UI.
7. **Small cleanups:** Vite also bundles its own copy of the 14.8 MB ORT wasm in `dist/` (we serve one from
   `public/ort/`); trim the duplicate before deploying.
