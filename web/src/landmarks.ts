// Video frames -> landmarks in the Kaggle 543-point layout, via @mediapipe/tasks-vision.
// Shared with the extension (imported as @shared/landmarks).

export const POINTS_PER_FRAME = 543

/** One frame: [x, y, z] for each of the 543 Kaggle points, NaN where a part was not detected. */
export type KaggleFrame = Float32Array

export function emptyFrame(): KaggleFrame {
  return new Float32Array(POINTS_PER_FRAME * 3).fill(Number.NaN)
}

// TODO: HandLandmarker (2 hands), PoseLandmarker, FaceLandmarker (drop the 10 iris points), and assign
// hands to slots by nearest pose wrist per feature_spec.hand_assignment.
