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

## Chosen points (89 total)

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

**Coordinates: x, y only.** z is dropped: it is noisy, and its scale differs between Holistic and the
separate web landmarkers, which would break parity.

## Processing order (both implementations must follow it exactly)

1. Select the 89 points, giving `(T, 89, 2)`, NaN where missing.
2. (Browser only) put detected hands in slots by nearest pose wrist, see below.
3. Presence masks per frame for `left_hand`, `right_hand`, `lips`, `pose`.
4. Trim leading and trailing frames that have no hand.
5. Mirror if the left-hand slot is present in more frames than the right: `x = 1 - x`, swap hand slots,
   swap left/right pose pairs and lip pairs (`mirroring.swap_pairs`). Hand indices need no reordering.
   After this, the dominant hand is always in the `right_hand` slot.
6. Normalize: center on mid-shoulder, divide by shoulder width. Both are **per-sequence** means, not
   per-frame, so shoulder jitter does not shake the whole skeleton.
7. Resample to 32 frames by nearest index (no interpolation, so a missing hand never gets blended with a
   present one, and Python and JS produce bit-identical indices).
8. NaN to 0.
9. Flatten to `(32, 182)`: 89 points as interleaved x, y (178), then the 4 masks.

## Open questions to check against the real data (Sat 6 PM milestone)

- **Hand slots.** Do Kaggle `left_hand` points sit near pose 15 (subject's left wrist)? Measure the mean
  distance from hand point 0 to pose 15 vs. pose 16 per slot. If it is the other way round, swap
  `hand_assignment` wrists and set `verified_against_kaggle: true`. The web `HandLandmarker` handedness label
  assumes a mirrored selfie image, so we ignore it on purpose.
- **Lip mirror pairs.** Believed correct from mesh symmetry. Verify by mirroring a real frame and checking
  each pair's x values swap within ~0.01.
- **Aspect ratio.** x and y are normalized to width and height separately, so shoulder-width scaling is
  not isotropic. If most Kaggle clips are portrait phone video and the browser sees landscape webcam/YouTube
  frames, hand positions will be stretched. Estimate the Kaggle aspect ratio (e.g. from the face-mesh
  eye-distance vs. eye-to-chin ratio) and, if needed, center-crop browser frames to match before landmarking.
- **Resample length.** 32 assumes clips of roughly 0.5 to 3 s. Check the Kaggle frame-count histogram after
  trimming; switch to 64 if many sequences are longer than ~60 frames.
- **Fallback scale.** Replace the 0.35 placeholder with the median shoulder width in the training data.
