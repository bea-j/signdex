// Landmark frames -> model input, the exact transform of model/prepare.py (FeatureExtractor.compute),
// driven by model/feature_spec.json. test/parity.test.ts checks both agree within 1e-4.
//
//   stage1: select points -> trim hand-less ends -> mirror if left-dominant -> normalize  (NaN kept)
//   stage2: resample to L frames (nearest) -> presence masks -> NaN to fill value           -> (L, F)
//
// Arithmetic mirrors numpy's float32 steps (Math.fround) so results are usually bit-identical.
import { COORDS_PER_POINT, type KaggleFrame } from './layout'
import { featureSpec as spec } from './spec'

const groups = spec.groups
/**
 * The spec's resample length. The model in use may have been trained with a different L (its ONNX input is
 * [batch, L, F]); pass that L to computeFeatures/stage2. v3 models use 32, v4 models 64.
 */
export const SEQUENCE_LENGTH: number = spec.resample.length
export const NUM_POINTS: number = spec.num_points
const MASKS: string[] = spec.presence.mask_order
export const FEATURES_PER_FRAME: number = NUM_POINTS * spec.coords.length + MASKS.length
if (FEATURES_PER_FRAME !== spec.output.shape[1] || SEQUENCE_LENGTH !== spec.output.shape[0]) {
  throw new Error('feature_spec.json output.shape does not match its groups/masks')
}
const N = NUM_POINTS
const FILL: number = spec.missing.fill_value

const GIDX = Int32Array.from(groups.flatMap((g) => g.global_index))
if (GIDX.length !== N) throw new Error('feature_spec.json num_points does not match groups')

const slices = new Map<string, [number, number]>()
const local = new Map<string, Map<number, number>>() // group -> landmark_index -> position among the N points
const typeGroup = new Map<string, string>()
{
  let o = 0
  for (const g of groups) {
    const k = g.global_index.length
    slices.set(g.name, [o, o + k])
    local.set(g.name, new Map(g.landmark_index.map((li, j) => [li, o + j])))
    if (!typeGroup.has(g.type)) typeGroup.set(g.type, g.name)
    o += k
  }
}

function pos(group: string, landmark: number): number {
  const p = local.get(group)?.get(landmark)
  if (p === undefined) throw new Error(`landmark ${landmark} not selected in group ${group}`)
  return p
}

const MIRROR: Int32Array = (() => {
  const perm = Int32Array.from({ length: N }, (_, i) => i)
  for (const [a, b] of spec.mirroring.swap_groups) {
    const [sa, ea] = slices.get(a)!
    const [sb, eb] = slices.get(b)!
    if (ea - sa !== eb - sb) throw new Error(`cannot swap ${a} and ${b}: sizes differ`)
    for (let j = 0; j < ea - sa; j++) {
      perm[sa + j] = sb + j
      perm[sb + j] = sa + j
    }
  }
  for (const [name, pairs] of Object.entries(spec.mirroring.swap_pairs)) {
    if (name.startsWith('_')) continue
    for (const [a, b] of pairs as number[][]) {
      const pa = pos(name, a)
      const pb = pos(name, b)
      perm[pa] = pb
      perm[pb] = pa
    }
  }
  return perm
})()

const norm = spec.normalization
const CENTER = norm.center.landmark_index.map((i) => pos(typeGroup.get(norm.center.type)!, i))
const SCALE = norm.scale.landmark_index.map((i) => pos(typeGroup.get(norm.scale.type)!, i))
const FALLBACK_C = norm.fallback_if_no_shoulders.center
const FALLBACK_S = norm.fallback_if_no_shoulders.scale
const EPS = norm.eps
const LH = MASKS.indexOf('left_hand')
const RH = MASKS.indexOf('right_hand')

/** Stage-1 output: T frames x N points x (x, y), NaN = missing. */
export interface Stage1 {
  T: number
  x: Float32Array
  mirrored: boolean
}

function presence(x: Float32Array, T: number): Uint8Array {
  const out = new Uint8Array(T * MASKS.length)
  for (let t = 0; t < T; t++) {
    for (let m = 0; m < MASKS.length; m++) {
      const [s, e] = slices.get(MASKS[m])!
      for (let p = s; p < e; p++) {
        if (!Number.isNaN(x[(t * N + p) * 2])) {
          out[t * MASKS.length + m] = 1
          break
        }
      }
    }
  }
  return out
}

export function stage1(frames: KaggleFrame[]): Stage1 {
  const T0 = frames.length
  let x = new Float32Array(T0 * N * 2)
  for (let t = 0; t < T0; t++) {
    const f = frames[t]
    for (let p = 0; p < N; p++) {
      const src = GIDX[p] * COORDS_PER_POINT
      x[(t * N + p) * 2] = f[src]
      x[(t * N + p) * 2 + 1] = f[src + 1]
    }
  }
  if (T0 === 0) return { T: 0, x, mirrored: false }

  let pres = presence(x, T0)
  const M = MASKS.length
  let first = -1
  let last = -1
  for (let t = 0; t < T0; t++) {
    if (pres[t * M + LH] || pres[t * M + RH]) {
      if (first < 0) first = t
      last = t
    }
  }
  let T = T0
  if (first >= 0) {
    x = x.slice(first * N * 2, (last + 1) * N * 2)
    pres = pres.slice(first * M, (last + 1) * M)
    T = last - first + 1
  }

  let nL = 0
  let nR = 0
  for (let t = 0; t < T; t++) {
    nL += pres[t * M + LH]
    nR += pres[t * M + RH]
  }
  const mirrored = nL > nR
  if (mirrored) {
    const y = new Float32Array(x.length)
    for (let t = 0; t < T; t++) {
      for (let p = 0; p < N; p++) {
        const q = (t * N + MIRROR[p]) * 2
        y[(t * N + p) * 2] = 1 - x[q] // stored as float32 == numpy's float32 1.0 - x
        y[(t * N + p) * 2 + 1] = x[q + 1]
      }
    }
    x = y
  }

  // per-sequence center (mean mid-shoulder) and scale (mean shoulder distance), in float64
  let cx = 0
  let cy = 0
  let sd = 0
  let nValid = 0
  for (let t = 0; t < T; t++) {
    const a = (t * N + CENTER[0]) * 2
    const b = (t * N + CENTER[1]) * 2
    const mx = (x[a] + x[b]) / 2
    const my = (x[a + 1] + x[b + 1]) / 2
    const sa = (t * N + SCALE[0]) * 2
    const sb = (t * N + SCALE[1]) * 2
    const d = Math.sqrt((x[sa] - x[sb]) ** 2 + (x[sa + 1] - x[sb + 1]) ** 2)
    if (Number.isFinite(mx) && Number.isFinite(my) && Number.isFinite(d)) {
      cx += mx
      cy += my
      sd += d
      nValid++
    }
  }
  let s: number
  if (nValid > 0) {
    cx /= nValid
    cy /= nValid
    s = sd / nValid
  } else {
    ;[cx, cy] = FALLBACK_C
    s = FALLBACK_S
  }
  const c32x = Math.fround(cx)
  const c32y = Math.fround(cy)
  const s32 = Math.fround(Math.max(s, EPS))
  for (let i = 0; i < x.length; i += 2) {
    x[i] = Math.fround(x[i] - c32x) / s32
    x[i + 1] = Math.fround(x[i + 1] - c32y) / s32
  }
  return { T, x, mirrored }
}

/**
 * Drop `start` frames from the beginning and `end` from the end of a stage-1 sequence (keeps at least 1 frame).
 * Used for test-time averaging; it is not part of the spec transform, so it doesn't affect parity.
 */
export function trimStage1(s1: Stage1, start: number, end: number): Stage1 {
  if (s1.T === 0) return s1
  const a = Math.max(0, Math.min(start, s1.T - 1))
  const b = Math.max(0, Math.min(end, s1.T - 1 - a))
  return { T: s1.T - a - b, x: s1.x.subarray(a * N * 2, (s1.T - b) * N * 2), mirrored: s1.mirrored }
}

export function resampleIndex(T: number, L = SEQUENCE_LENGTH): Int32Array {
  const idx = new Int32Array(L)
  for (let i = 0; i < L; i++) idx[i] = Math.min(T - 1, Math.floor(((2 * i + 1) * T) / (2 * L)))
  return idx
}

/** Returns Float32Array of length L * FEATURES_PER_FRAME, row-major (frame, feature). */
export function stage2(s1: Stage1, L: number = SEQUENCE_LENGTH): Float32Array {
  if (!Number.isInteger(L) || L < 1) throw new Error(`invalid resample length ${L}`)
  const F = FEATURES_PER_FRAME
  const out = new Float32Array(L * F)
  if (s1.T === 0) return out
  const idx = resampleIndex(s1.T, L)
  const sel = new Float32Array(L * N * 2)
  for (let i = 0; i < L; i++) sel.set(s1.x.subarray(idx[i] * N * 2, (idx[i] + 1) * N * 2), i * N * 2)
  const masks = presence(sel, L)
  for (let i = 0; i < L; i++) {
    for (let k = 0; k < N * 2; k++) {
      const v = sel[i * N * 2 + k]
      out[i * F + k] = Number.isNaN(v) ? FILL : v
    }
    for (let m = 0; m < MASKS.length; m++) out[i * F + N * 2 + m] = masks[i * MASKS.length + m]
  }
  return out
}

/** L: resample length of the model in use (its ONNX input [batch, L, F]); defaults to the spec's. */
export function computeFeatures(frames: KaggleFrame[], L: number = SEQUENCE_LENGTH): Float32Array {
  return stage2(stage1(frames), L)
}
