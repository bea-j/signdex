# feature_spec.json, explained

`feature_spec.json` is the single source of truth for turning landmarks into model input. Both
`model/prepare.py` (training) and `web/src/features.ts` (browser) read it. If you change it, bump `version`
and retrain. The parity test (CLAUDE.md) must pass after every change.

## Kaggle 543-point layout

Every frame has 543 rows in this order. `global_index = type_offset + landmark_index`.

| type         | landmark_index | global_index |
|--------------|----------------|--------------|
| `face`       | 0 to 467       | 0 to 467     |
| `left_hand`  | 0 to 20        | 468 to 488   |
| `pose`       | 0 to 32        | 489 to 521   |
| `right_hand` | 0 to 20        | 522 to 542   |

## Chosen points (109 total)

**Hands: all 21 points each (42).** Handshape is most of the signal. Hand indices are anatomical
(0 wrist, 4 thumb tip, 8 index tip, ...), same in Holistic and the web `HandLandmarker`.

**Lips: 40 face-mesh points.** Outer and inner lip contours from MediaPipe's `FACEMESH_LIPS`, deduplicated:
- upper outer: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
- lower outer: 146, 91, 181, 84, 17, 314, 405, 321, 375
- upper inner: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415
- lower inner: 95, 88, 178, 87, 14, 317, 402, 318, 324, 308

Mouth shape and mouthing carry meaning for many signs. All indices are below 468, so they are identical in
the web `FaceLandmarker` output (478 points; the last 10 are iris points, which we never use).
This is the 40-point list used by several top public Kaggle solutions.

**Pose: 7 upper-body points.** 0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists.
- Shoulders give the normalization frame and body-relative hand location (chest vs. chin vs. forehead).
- Elbows and wrists give arm movement, and the wrists still locate a hand when `HandLandmarker` loses it
  to motion blur.
- Dropped: pose 1 to 10 (eyes, ears, mouth corners; redundant with lips and nose), pose hand points 17 to 22 (redundant with
  hand landmarks), hips and legs 23 to 32 (often out of frame in phone and webcam shots, so they would be
  noise at inference time). CLAUDE.md said about 8; 7 is the set that survives this cut.

**Upper face: 20 face-mesh points (added in spec v3).** Many signs differ mainly by where the hand
touches the face: forehead (KNOW, THINK, FATHER), eyes (SEE, CRY), chin (MOTHER). Lips and nose alone
anchor only the lower face. These points let the model relate hand position to the upper face directly;
several top Kaggle solutions also used eye contours.
- each eye: outer corner, inner corner, upper-lid center, lower-lid center: 33, 133, 159, 145 (subject's
  right eye) and 263, 362, 386, 374 (left eye)
- each eyebrow, upper line, outer to inner: 70, 63, 105, 66, 107 (right) and 300, 293, 334, 296, 336 (left)
- midline: 10 (top of forehead) and 152 (chin)
- mirror pairs follow the same order (33↔263, …, 107↔336); 10 and 152 map to themselves.
- No separate presence mask: these come from the same face detection as the lips.
- All indices are below 468, so they exist unchanged in the web `FaceLandmarker` output. `train.py`
  checks the mirror pairs on real Kaggle frames before training and stops if any pair is not left/right
  symmetric.

**Coordinates: x, y only.** z is dropped: it is noisy, and its scale differs between Holistic and the
separate web landmarkers, which would break parity.

## Processing order (both implementations must follow it exactly)

0. (Browser only) rescale raw y for the frame's aspect ratio: `y' = 0.5 + (y - 0.5) * (H/W) / 1.25`
   (`input_aspect`). Training data is used as-is.
1. Select the 109 points, giving `(T, 109, 2)`, NaN where missing.
2. (Browser only) put detected hands in slots by nearest pose wrist (`left_hand` ↔ pose 15,
   `right_hand` ↔ pose 16).
3. Presence masks per frame for `left_hand`, `right_hand`, `lips`, `pose`.
4. Trim leading and trailing frames that have no hand.
5. Mirror if the left-hand slot is present in more frames than the right: `x = 1 - x`, swap hand slots,
   swap left/right pose, lip and upper-face pairs (`mirroring.swap_pairs`). Hand indices need no reordering.
   After this, the dominant hand is always in the `right_hand` slot.
6. Normalize: center on mid-shoulder, divide by shoulder width. Both are **per-sequence** means, not
   per-frame, so shoulder jitter does not shake the whole skeleton.
7. Resample to L = 64 frames by nearest index, `src = min(T-1, ((2i+1)·T) // (2L))` in integer
   arithmetic (no interpolation, so a missing hand never gets blended with a present one, and Python and
   JS pick identical frames).
8. NaN to 0.
9. Flatten to `(64, 222)`: 109 points as interleaved x, y (218), then the 4 masks.

**Frame length and the model in use (v4).** v4 changed only L, from 32 to 64; every per-frame value is
the same as in v3. The model's ONNX input is `[batch, L, 222]`. `features.ts` (`computeFeatures(frames,
L)`) and `prepare.py` (`FeatureExtractor(spec, length=L)`) take L as a parameter, so the app should
pass the L declared by the `model.onnx` it loads. The current v3 model (L = 32) keeps working until a
v4 model replaces it. The parity test checks both lengths.

**Velocity is computed inside the v4 model,** not in the features. The model derives frame-to-frame
differences `x[t+1] − x[t]` and `x[t+2] − x[t]` from its input, zeroed wherever either frame is missing
(using the presence masks). So the browser computes the same per-frame features as before.

## Data check results (spec v2)

Measured on 500 random training sequences by `kaggle/datacheck/datacheck.py`; the full numbers are in
`kaggle/datacheck/output/datacheck.json`. All 500 files had the exact 543-row face / left_hand / pose /
right_hand order, and their frame numbers were strictly increasing.

- **Hand slots: verified.** Kaggle `left_hand` is nearer pose wrist 15 in 99.9% of frames (median distance
  0.05 vs 0.92), and `right_hand` is nearer pose 16. The videos are not mirrored: the subject's left side
  appears on the image's right. The browser assigns hands by nearest pose wrist. It uses the
  `HandLandmarker` handedness label only when no pose is detected, because that label's meaning has
  changed between versions (tasks-vision 1.2.3 documents it as the subject's hand for an unmirrored image).
- **Mirroring matters.** 41% of sequences have the left-hand slot present in more frames than the right.
- **Aspect ratio: about 3:4 portrait to square, not landscape.** Palm bones are rigid, and MediaPipe hand
  z is in x units. So `sqrt(dx² + (a·dy)² + dz²)` should stay constant as the hand rotates only when `a`
  equals the true H/W. Scanning `a` gives a minimum at 1.20 (1.36 using xy only). Scores: 3:4 portrait 0.104,
  1:1 0.107, 4:3 landscape 0.122, 16:9 landscape 0.135. The per-sequence estimates are spread out, so the
  data probably mixes aspects. We use **1.25**. A 16:9 webcam frame (H/W 0.56) would be a 2.2× y-vs-x
  distortion, far beyond what augmentation can absorb, hence the `input_aspect` correction in the browser.
  Training also uses ±15% x/y stretch to cover the remaining uncertainty.
- **Shoulder width:** median 0.56 of image width, so subjects are framed tightly. Shoulders were present
  in 100% of frames; the fallback is now center (0.49, 0.65), scale 0.56.
- **Frame counts (after trim):** median 23, p75 45, p95 129; 33% are over 32 frames and 13% over 64.
  v2/v3 used 32 frames. v4 uses 64, as the top Kaggle ISLR solutions use longer windows: 87% of signs
  are then never downsampled, and the longer window gives the in-model velocity features finer timing.
- **Lip mirror pairs: verified.** Each pair's midpoint is within 1.5% of mouth width of the face midline.
- **Presence:** lips 99%, pose 100%, left hand 28% and right hand 38% of frames (hands are often
  undetected, which is why the masks exist).
