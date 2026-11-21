# SignSeek: look up ASL signs you see in videos

Hackathon project for WHACK 2026 (Wellesley, Nov 21 to 22). Hacking ends **Sunday 12:00 PM**. Judging 12:30 PM.
Track: Better Worlds (accessibility, education). Name is a placeholder.

## One-liner
Google's new sign-to-text helps signers type on Pixel 11. SignSeek helps anyone learning ASL understand the
signs they see in other people's videos, on any device, right in the browser.

## The problem
ASL learners (college ASL students, hearing parents of Deaf kids, interpreters in training) constantly hit signs
they don't recognize while watching Deaf creators, class recordings, or YouTube lessons. Existing ASL dictionaries
are searched by English word, which is useless when you don't know the word. Sign-to-text tools (Gboard on
Pixel 11, small translator apps) watch *you* signing live into your own camera. Nothing lets you point at a moment
in someone else's video and get a dictionary result.

## What we are building (in priority order)
1. **Web app (MUST, this is the guaranteed demo)**: upload a clip or use webcam, trim a 1 to 3 s window, get top 5
   sign matches with confidence, reference links, and "looks similar to" alternatives.
2. **Chrome extension (SHOULD)**: while watching a video in Chrome, press a hotkey; the extension uses the last few
   seconds of the video, lets the user trim to the sign, and shows results in a side panel.
3. **Word list + quiz (NICE)**: save looked-up signs; quiz mode replays the clip and asks for the meaning.

Not building: sentence translation, continuous signing, sign generation/avatars. Do not call it a "translator".

## Data and model
- Vocabulary: the **250 signs** from Google's Kaggle "Isolated Sign Language Recognition" competition
  (`kaggle competitions download -c asl-signs`). Data is MediaPipe Holistic landmarks per frame, already extracted.
  Underlying videos are PopSign ASL v1.0 (CC BY 4.0). Check Kaggle competition rules on reuse; credit both in the UI.
- Landmark layout in the Kaggle parquet files: per frame, 543 points = face (468) + left_hand (21) + pose (33) +
  right_hand (21); columns `frame, type, landmark_index, x, y, z`. x, y normalized to image size.
- **Parity rule (most important technical decision):** train only on landmarks we can reproduce exactly in the
  browser. Use **both hands (42) + lips subset of face (~40) + upper-body pose (~8)**. Drop z (noisy) unless it helps.
  Write ONE shared spec `model/feature_spec.json` (landmark indices, ordering, normalization) used by BOTH the
  Python training code and the JS inference code.
- Normalization: center on mid-shoulder (or nose), scale by shoulder width; handle missing hands (NaN -> 0 plus a
  mask); handle left-handed signers by mirroring when the left hand is dominant; resample to a fixed length
  (e.g. 32 or 64 frames).
- Model: small 1D-CNN or transformer over the frame sequence (public Kaggle solutions are good references).
  Train on a Kaggle/Colab GPU. Target: >70% top-1, >90% top-5 on held-out signers. Output softmax over 250.
- Export: TF -> TFLite or PyTorch -> ONNX. Run in the browser with ONNX Runtime Web or TF.js.
- "Similar signs": compute from the validation confusion matrix (signs most often confused with this one), saved as
  `similar.json`.

## Browser pipeline
- MediaPipe Tasks for Web (`@mediapipe/tasks-vision`): `HandLandmarker` (2 hands), `PoseLandmarker`,
  `FaceLandmarker` (478 points; the first 468 match the Kaggle face mesh indices, drop the 10 iris points).
  If `HolisticLandmarker` works on web, it may be simpler; verify the output matches the training layout.
- Video frames -> landmarks at ~15 to 30 fps -> keep a rolling buffer of the last ~6 s of landmarks plus small
  thumbnails for the scrubber -> on lookup, slice the chosen window -> features via feature_spec -> model ->
  top 5.
- Everything runs on device. No video leaves the browser. Say this in the UI and the pitch.

## Chrome extension notes (Manifest V3)
- Preferred capture: `chrome.tabCapture` stream rendered in an **offscreen document**, frames drawn to canvas.
- Fallback: content script draws the page's `<video>` element to a canvas; may fail on some sites if the canvas is
  tainted. Detect and fall back to tabCapture.
- UI in `chrome.sidePanel`. Hotkey via `commands` (e.g. Alt+Shift+S). Word list in `chrome.storage.local`.

## Results panel content
For each match: gloss (e.g. "THANK-YOU"), English meaning, confidence bar, a link to look it up on an external
dictionary (search URL to signasl.org or handspeak.com; do not copy their videos), and "looks similar to" chips.
If top-1 confidence < 0.4, say "Not sure. Try trimming tighter around the sign."

## Repo layout
```
signseek/
  CLAUDE.md
  model/            # Python: data prep, training, eval, export
    feature_spec.json
    prepare.py  train.py  evaluate.py  export.py
    labels.json  similar.json
  web/              # Vite + TypeScript web app (upload, webcam, trim, results)
    src/landmarks.ts  src/features.ts  src/model.ts  src/ui/...
    public/model.onnx
  extension/        # MV3 extension, reuses web/src/{landmarks,features,model}.ts
  demo/             # pre-tested demo clips + expected results
```

## Conventions
- TypeScript, Vite, no heavy UI framework required (plain TS or React, team's choice; keep it small).
- `features.ts` and `prepare.py` must have a **parity test**: run both on the same exported landmark JSON and
  assert the feature tensors match within 1e-4.
- Keep accessibility basics: keyboard operable, labeled buttons, captions on reference clips, good contrast.

## Milestones (clock time)
- **Sat 6 PM**: data downloaded; feature_spec.json agreed; web app shows live landmarks on webcam.
- **Sat 10 PM**: baseline model trained and exported; parity test passes.
- **Sun 2 AM**: end to end lookup works in the web app on uploaded clips.
- **Sun 6 AM**: extension capture working (or decision to demo web app only).
- **Sun 9 AM**: similar signs, word list, polish, 3 to 5 pre-tested demo clips.
- **Sun 11 AM**: Devpost written, demo video recorded, feature freeze.

## Demo script (90 s)
1. Problem: learner watching a Deaf creator misses a sign; dictionaries need the English word.
2. Press the hotkey on a YouTube clip, trim, result appears with similar signs.
3. Live: a judge or teammate signs into the webcam; lookup works.
4. Close: on device, any browser, built for learners; credit PopSign/Kaggle data and community framing.
