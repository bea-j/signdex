// One sign lookup: aspect-corrected landmark frames -> features (at the model's frame length) -> model -> top k,
// with optional test-time averaging over trimmed variants of the clip. Shared by the debug page and the app.
import { stage1, stage2, trimStage1, type Stage1 } from './features'
import { COORDS_PER_POINT, OFFSETS, type KaggleFrame } from './layout'
import type { SignMatch, SignModel } from './model'

export interface LookupResult {
  matches: SignMatch[]
  /** Stage-1 of the whole clip (frames with hands, mirrored or not). */
  stage1: Stage1
  variants: number
  /** Frames (of the input) with at least one hand. 0 means the matches are meaningless. */
  handFrames: number
  timing: { featuresMs: number; modelMs: number; totalMs: number }
}

/**
 * Start/end offsets (frames) for test-time averaging: the original, then the start and/or end moved inward by
 * d ~ 6% of the clip (2-3 frames at 20 fps). Applied after stage 1 (where training augmentation trims too), so
 * each variant really differs. Very short clips get just the original.
 */
export function ttaOffsets(T: number, n: number): [number, number][] {
  const all: [number, number][] = [[0, 0]]
  if (n <= 1 || T < 8) return all
  const d = Math.max(1, Math.round(0.06 * T))
  all.push([d, 0], [0, d], [2 * d, 0], [0, 2 * d])
  return all.slice(0, Math.min(n, all.length))
}

/** Number of frames with at least one detected hand landmark. */
export function countHandFrames(frames: KaggleFrame[]): number {
  let n = 0
  for (const f of frames) {
    let found = false
    for (const off of [OFFSETS.left_hand, OFFSETS.right_hand]) {
      for (let i = 0; i < 21 && !found; i++) if (!Number.isNaN(f[(off + i) * COORDS_PER_POINT])) found = true
    }
    if (found) n++
  }
  return n
}

export async function lookupSign(
  model: SignModel,
  frames: KaggleFrame[],
  opts: { variants?: number; k?: number } = {},
): Promise<LookupResult> {
  const t0 = performance.now()
  const s1 = stage1(frames)
  const inputs = ttaOffsets(s1.T, opts.variants ?? 1).map(([a, b]) => stage2(trimStage1(s1, a, b), model.length))
  const t1 = performance.now()
  const matches = await model.topKAveraged(inputs, opts.k ?? 5)
  const t2 = performance.now()
  return {
    matches,
    stage1: s1,
    variants: inputs.length,
    handFrames: countHandFrames(frames),
    timing: { featuresMs: t1 - t0, modelMs: t2 - t1, totalMs: t2 - t0 },
  }
}
