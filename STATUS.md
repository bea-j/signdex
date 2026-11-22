# Signdex status

_Updated Sun Nov 22, 4:55 PM ET. Submitted and judged (hacking ended Sun 12:00 PM); the live site can change._

**Live:** https://signdex.study (release `20261122-040825`, deployed Sat Nov 21, 11:08 PM ET), debug page at
https://signdex.study/debug.html. Add `?debug=1` to any page URL for an on-screen diagnostics panel (phones).
Everything in git is pushed to `main`. Redeploy with `bash deploy/deploy.sh --yes build upload verify`.

## What's live

- **Home:** "Practice American Sign Language with ease", a looping demo panel (labeled DEMO, fixed aspect,
  reference skeletons for hat/cat/drink/yes), three how-it-works steps with arrows, and links to the guide.
- **Webcam and uploaded-clip screens:** camera or player on the left, top 5 on the right (below on phones),
  updated in place. Webcam: Space/R, 3-2-1 countdown, 2 s recording, replay of the clip. Upload: drag a 1–3 s
  window. Switches (remembered): hand & body points, face points, details panel.
- **Phones (iOS Safari / Chrome on iOS):** Record shows loading progress and offers a retry; ORT's wasm memory
  is capped for iOS; MediaPipe falls back to CPU if the GPU fails at setup or later; the camera preview keeps
  its real shape (portrait too); touch wording instead of key hints; 56 px Record pinned under the camera;
  44 px tap targets. Tested in Chrome with iPhone emulation and in WebKit (engine only); not yet on a real iPhone.
- **Result cards:** display name and meaning (`web/public/dictionary.json`, built from `demo/dictionary.csv`
  with `npm run dictionary -w web`), confidence colors, "Looks similar to", "Watch a real signer", Save, and a
  reference animation, or a "Two-handed sign" tile (see below).
- **Dictionary page:** all 250 signs, searchable by name, animations only for cards on screen, pause switch.
- **My signs + quiz:** meanings and usage only, never how a sign is made. The Worker (`POST {"savedWords":
  [...]}`, 15 most recent) is asked first, but only its meaning/usage questions are shown (`quiz-content.ts`),
  with explanations from `dictionary.json`; every sign it leaves out gets a dictionary meaning question.
  Right now the Worker returns formation questions only (25 of 25 sampled), so quizzes are all dictionary
  questions. The whole quiz comes from the dictionary when the Worker is unreachable.
- **Guide ("How it works") and Privacy pages.** Signdex itself sends only saved sign names (for the quiz).
  MediaPipe also sends small usage pings to Google (`odml.pa.googleapis.com`); the pages don't mention that yet.
- **Animated background** (p5 sketch, self-hosted): header switch, off by default on phones/low-power devices,
  paused on the webcam screen and during clip analysis. It measured no lookup slowdown on this laptop.

## Model in use: v4 (69.0% top-1, 83.7% top-5)

Held-out validation: 4 participants (4718, 28656, 36257, 62590), 17,521 sequences. From the
`signseek-evaluate` Kaggle kernel:

| Model | Frames | top-1 | top-5 | with 5-version averaging |
|---|---|---|---|---|
| v2 (previous) | 32 | 65.5% | 82.3% | 65.7% / 82.5% |
| **v4 (in use)** | **64** | **69.0%** | **83.7%** | 68.9% / 83.4% |
| v5 | 64 | 68.9% | 83.7% | 68.7% / 83.5% |
| v4 + v5 ensemble | 64 | 69.8% | 84.6% | 69.6% / 84.3% (not used: < +1 point, 2x size) |

Model files are **not in git**: `model/{model.onnx, labels.json, similar.json, metrics.json,
parity_fixture.json, training.log}`. v2 is backed up in `model/runs/v2-backup/`.
Teammates: unzip `teammate-assets.zip` (repo root, gitignored) into the repo root; its README lists every file.

## Reference animations

- `model/reference_export.py` (Kaggle CPU kernel `beatricej/signseek-reference`, `python kaggle/bundle.py
  --entry reference`): scores all 76,956 training-signer recordings with v4 and picks, per label, a correctly
  classified clip (>= 12 frames) with the best hand visibility, then confidence. Output `signs.bin` +
  `shortlist.npz` (local copy in `model/runs/reference-kernel/`, not in git).
- **Two-handed signs:** in this data the second hand is tracked in only ~6% of frames (likely phone-recorded).
  `TWO_HANDED_SIGNS` in `model/reference.py` (71 signs) get an animation only from a clip with both hands in
  >= 50% of its frames and >= 12 frames. None qualifies, so all 71 are **withheld** (`python
  model/reference.py withhold`) and show a "Two-handed sign / Watch a real signer" tile. 179 signs animate.
- Withheld: all, alligator, animal, arm, awake, balloon, bath, before, beside, boat, book, brother, can, car,
  chair, chocolate, clean, close, cloud, dance, drawer, empty, every, fall, fast, finger, finish, first, flag,
  gift, hate, have, helicopter, hide, into, jacket, jump, make, many, mitten, morning, night, now, on, open,
  outside, owl, pen, pencil, person, puzzle, quiet, rain, read, ride, room, sad, shoe, sick, snow, store,
  story, table, tiger, time, touch, tree, wait, wake, wet, zebra.

## Chrome extension

`npm run build -w extension`, load `extension/dist` unpacked. Alt+Shift+S starts capture; press again (or
"Look up last N s", 1–5 s) to look up. Same model, cards, dictionary and animations as the web app.

## Not verified yet

- A real webcam lookup in the current layout (headless tests used recorded Kaggle landmark sequences).
- The extension's second-hotkey lookup in real Chrome (Chrome no longer loads unpacked extensions from the
  command line).

## Next steps

1. **Two-handed animations (kernel/data):** the training signers have no usable two-handed clips. Options:
   (a) re-run `reference_export.py` including the 4 validation signers' recordings (small chance of a
   both-hands clip); (b) record our own reference clips for the 71 signs with a fluent signer on the debug
   page ("Download last 6 s as JSON") and add a small converter from those recordings to `signs.bin` blocks
   (same stage 1 as `features.ts`); (c) keep the tiles. Also have a fluent signer review `TWO_HANDED_SIGNS`.
2. **Try it for real** in Chrome: webcam lookups on signdex.study, an uploaded clip, and the extension on YouTube.
3. **Quiz content (ph105):** the Worker's prompt should ask only for meaning and usage questions (what a sign
   means, which sign fits a situation or sentence), never handshape, movement, location, orientation, facial
   expression or comparisons between signs. Until then the app filters (see above). DRINK is still never sent
   (`WORKER_EXCLUDED_SIGNS` in `web/src/app/quiz.ts`).
4. **Test-time averaging:** doesn't help v4 and costs 5x model time; set `TTA_VARIANTS = 1` in
   `web/src/app/webcam.ts` and `trim.ts` if lookups feel slow.
5. **Refresh local copies:** `teammate-assets.zip` was rebuilt Sun Nov 22 with the current `signs.bin`
   (`5871be8caaff`, 179 animated + 71 withheld). Still to do: rebuild the extension (`npm run build -w
   extension`) to pick up the withheld animations.
6. **Final model:** retrain v4 on all 21 participants for the demo build (no held-out score afterwards).
7. **Demo:** 3–5 pre-tested clips in `demo/` with expected results.
8. **Open questions:** the home-page list started at item 2 (item 1 never arrived); no `design/wallpaper.*`
   was provided; animated background is on by default on desktop (one line to make it off everywhere);
   `design/figma-make/` is not in git (its `.gitattributes` routes images through Git LFS).
