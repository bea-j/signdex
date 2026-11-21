// The Kaggle 543-point frame layout and JSON (de)serialization. No MediaPipe import, so the parity test,
// features.ts and the extension can use it anywhere.
import { featureSpec } from './spec'

export const POINTS_PER_FRAME = featureSpec.source_layout.points_per_frame
export const COORDS_PER_POINT = 3 // x, y, z
export const OFFSETS = featureSpec.source_layout.type_offsets
export const SIZES = featureSpec.source_layout.type_sizes

/** One frame: [x, y, z] for each of the 543 Kaggle points, NaN where a part was not detected. */
export type KaggleFrame = Float32Array

export function emptyFrame(): KaggleFrame {
  return new Float32Array(POINTS_PER_FRAME * COORDS_PER_POINT).fill(Number.NaN)
}

/** A frame as JSON: 543 points of [x, y] or [x, y, z], null = NaN. */
export type NestedFrame = (number | null)[][]

export function frameFromNested(points: NestedFrame): KaggleFrame {
  if (points.length !== POINTS_PER_FRAME) throw new Error(`expected ${POINTS_PER_FRAME} points, got ${points.length}`)
  const f = emptyFrame()
  for (let p = 0; p < POINTS_PER_FRAME; p++) {
    const pt = points[p]
    for (let c = 0; c < Math.min(COORDS_PER_POINT, pt.length); c++) {
      const v = pt[c]
      f[p * COORDS_PER_POINT + c] = v === null ? Number.NaN : v
    }
  }
  return f
}

export function frameToNested(f: KaggleFrame, digits = 6): NestedFrame {
  const k = 10 ** digits
  const out: NestedFrame = []
  for (let p = 0; p < POINTS_PER_FRAME; p++) {
    const pt: (number | null)[] = []
    for (let c = 0; c < COORDS_PER_POINT; c++) {
      const v = f[p * COORDS_PER_POINT + c]
      pt.push(Number.isNaN(v) ? null : Math.round(v * k) / k)
    }
    out.push(pt)
  }
  return out
}

/** Landmark recording exported by the debug page and read by the parity test. */
export interface LandmarkRecording {
  format: 'signseek-landmarks-v1'
  source: string
  width: number
  height: number
  fps: number
  /** Media time of each frame, seconds. */
  t: number[]
  /** Raw MediaPipe coordinates (no aspect correction applied). */
  frames: NestedFrame[]
}
