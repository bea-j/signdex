// Feature tensor -> top-k sign matches, via ONNX Runtime Web and public/model.onnx.
// Shared with the extension.

export interface SignMatch {
  gloss: string
  confidence: number
}

/** Below this top-1 confidence the UI says "Not sure. Try trimming tighter around the sign." */
export const LOW_CONFIDENCE = 0.4

export async function predictTopK(features: Float32Array, k = 5): Promise<SignMatch[]> {
  void features
  void k
  throw new Error('predictTopK: not implemented yet')
}
