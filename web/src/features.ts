// Landmark frames -> model input tensor, following model/feature_spec.json exactly.
// Must match model/prepare.py within 1e-4 (parity test). Shared with the extension.
import { featureSpec } from './spec'
import type { KaggleFrame } from './landmarks'

export const SEQUENCE_LENGTH = featureSpec.resample.length
export const FEATURES_PER_FRAME = featureSpec.output.shape[1]

/** Returns a Float32Array of length SEQUENCE_LENGTH * FEATURES_PER_FRAME, row-major (frame, feature). */
export function computeFeatures(frames: KaggleFrame[]): Float32Array {
  void frames
  throw new Error('computeFeatures: not implemented yet')
}
