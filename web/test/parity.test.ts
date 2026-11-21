// Parity test: features.ts must match model/prepare.py within 1e-4 (CLAUDE.md).
//
// 1. Collects landmark sequences in the Kaggle 543-point layout:
//    - synthetic edge cases generated here (seeded), exported to JSON
//    - model/parity_fixture.json: real Kaggle validation sequences + features computed on Kaggle
//    - demo/landmarks/*.json: recordings saved from the debug page (real MediaPipe web output)
// 2. Runs them through prepare.py (model/parity.py) and through features.ts.
// 3. Asserts every feature value matches within 1e-4.
//
// Python: $SIGNSEEK_PYTHON, else model/.venv, else python on PATH (needs numpy).
import { execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { beforeAll, describe, expect, it } from 'vitest'
import { computeFeatures, FEATURES_PER_FRAME, SEQUENCE_LENGTH, stage1 } from '../src/features'
import { frameFromNested, OFFSETS, POINTS_PER_FRAME, type NestedFrame } from '../src/layout'
import { featureSpec } from '../src/spec'

const TOL = 1e-4
const REPO = fileURLToPath(new URL('../..', import.meta.url))

interface Case {
  name: string
  frames: NestedFrame[]
  /** Resample length L to compute at. */
  length?: number
  expected?: number[][] // features computed on Kaggle (fixture only)
  expectMirrored?: boolean
}

/** Spec L (v4: 64) and the v3 model's L (32), which stays in use until a v4 model beats it. */
const LENGTHS = [...new Set([SEQUENCE_LENGTH, 32])]
const atLengths = (cs: Case[]): Case[] =>
  cs.flatMap((c) => LENGTHS.map((L) => ({ ...c, name: `${c.name} [L=${L}]`, length: L })))

// ------------------------------------------------------------------ synthetic sequences

function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface SynthOpts {
  T: number
  left?: [number, number] // frames [from, to) where the left_hand slot is present
  right?: [number, number]
  shoulders?: 'all' | 'none' | 'some'
  lipGaps?: number[]
  noWrists?: boolean
}

function synth(seed: number, o: SynthOpts): NestedFrame[] {
  const r = mulberry32(seed)
  const base: [number, number, number][] = []
  for (let p = 0; p < POINTS_PER_FRAME; p++) base.push([0.3 + 0.4 * r(), 0.2 + 0.6 * r(), -0.1 + 0.2 * r()])
  // plausible body anchors: subject's left shoulder (pose 11) on the image right
  const setPose = (i: number, x: number, y: number) => (base[OFFSETS.pose + i] = [x, y, 0])
  setPose(11, 0.75, 0.66)
  setPose(12, 0.22, 0.64)
  setPose(15, 0.8, 0.55)
  setPose(16, 0.2, 0.5)
  const inRange = (t: number, rg?: [number, number]) => !!rg && t >= rg[0] && t < rg[1]
  const frames: NestedFrame[] = []
  for (let t = 0; t < o.T; t++) {
    const f: NestedFrame = []
    for (let p = 0; p < POINTS_PER_FRAME; p++) {
      const [bx, by, bz] = base[p]
      const wobble = 0.02 * Math.sin(t / 3 + p)
      f.push([bx + wobble + 0.005 * (r() - 0.5), by - wobble + 0.005 * (r() - 0.5), bz])
    }
    const blank = (from: number, n: number) => {
      for (let p = from; p < from + n; p++) f[p] = [null, null, null]
    }
    if (!inRange(t, o.left)) blank(OFFSETS.left_hand, 21)
    if (!inRange(t, o.right)) blank(OFFSETS.right_hand, 21)
    if (o.lipGaps?.includes(t)) blank(OFFSETS.face, 468)
    if (o.shoulders === 'none' || (o.shoulders === 'some' && t % 3 === 0)) {
      blank(OFFSETS.pose + 11, 1)
      if (t % 2 === 0) blank(OFFSETS.pose + 12, 1)
    }
    if (o.noWrists) {
      blank(OFFSETS.pose + 15, 1)
      blank(OFFSETS.pose + 16, 1)
    }
    frames.push(f)
  }
  return frames
}

function syntheticCases(): Case[] {
  return [
    { name: 'synthetic: right-dominant, 40 frames, lip gaps', frames: synth(1, { T: 40, right: [5, 35], left: [10, 20], lipGaps: [7, 8] }), expectMirrored: false },
    { name: 'synthetic: left-dominant (mirrored), 40 frames', frames: synth(2, { T: 40, left: [3, 36], right: [12, 18] }), expectMirrored: true },
    { name: 'synthetic: tie between hands (not mirrored)', frames: synth(3, { T: 30, left: [5, 15], right: [15, 25] }), expectMirrored: false },
    { name: 'synthetic: no hands at all (no trim)', frames: synth(4, { T: 25 }), expectMirrored: false },
    { name: 'synthetic: no shoulders (fallback center/scale)', frames: synth(5, { T: 30, right: [2, 28], shoulders: 'none' }) },
    { name: 'synthetic: shoulders missing in some frames', frames: synth(6, { T: 33, right: [0, 33], shoulders: 'some' }) },
    { name: 'synthetic: pose wrists missing', frames: synth(7, { T: 20, left: [0, 20], noWrists: true }) },
    { name: 'synthetic: long 150 frames (downsample)', frames: synth(8, { T: 150, right: [10, 140], left: [60, 90] }) },
    { name: 'synthetic: short 7 frames (upsample)', frames: synth(9, { T: 7, left: [1, 6] }) },
    { name: 'synthetic: single frame', frames: synth(10, { T: 1, right: [0, 1] }) },
    { name: 'synthetic: empty sequence', frames: [] },
  ]
}

// ------------------------------------------------------------------ real sequences

function fixtureCases(): Case[] {
  const path = join(REPO, 'model', 'parity_fixture.json')
  if (!existsSync(path)) return []
  const fx = JSON.parse(readFileSync(path, 'utf8')) as {
    spec_version: number
    cases: { sequence_id: number; sign: string; mirrored: boolean; raw_xy: NestedFrame[]; features: number[][] }[]
  }
  // Raw landmarks are always useful. Kaggle-computed features are compared only if the fixture's spec
  // version has the same per-frame transform (feature_spec.per_frame_identical_to_versions), at its own L.
  const same: number[] = [featureSpec.version, ...(featureSpec.per_frame_identical_to_versions ?? [])]
  const comparable = same.includes(fx.spec_version)
  return fx.cases.flatMap((c) => {
    const base = {
      name: `kaggle: sequence ${c.sequence_id} "${c.sign}"${c.mirrored ? ' (mirrored)' : ''} (fixture spec v${fx.spec_version})`,
      frames: c.raw_xy,
      expectMirrored: c.mirrored,
    }
    if (!comparable) return atLengths([base])
    const L = c.features.length
    return [{ ...base, name: `${base.name} [L=${L}, vs Kaggle]`, length: L, expected: c.features }]
  })
}

function recordingCases(): Case[] {
  const dir = join(REPO, 'demo', 'landmarks')
  if (!existsSync(dir)) return []
  return readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .map((f) => {
      const rec = JSON.parse(readFileSync(join(dir, f), 'utf8'))
      if (rec.format !== 'signseek-landmarks-v1') throw new Error(`${f}: not a signseek-landmarks-v1 recording`)
      return { name: `recording: ${f}`, frames: rec.frames as NestedFrame[] }
    })
}

// ------------------------------------------------------------------ run both sides

function python(): string {
  if (process.env.SIGNSEEK_PYTHON) return process.env.SIGNSEEK_PYTHON
  for (const p of [join(REPO, 'model', '.venv', 'Scripts', 'python.exe'), join(REPO, 'model', '.venv', 'bin', 'python')]) {
    if (existsSync(p)) return p
  }
  return process.platform === 'win32' ? 'python' : 'python3'
}

const cases: Case[] = [...atLengths(syntheticCases()), ...fixtureCases(), ...atLengths(recordingCases())]
const pyFeatures = new Map<string, number[][]>()

beforeAll(() => {
  const dir = mkdtempSync(join(tmpdir(), 'signseek-parity-'))
  const input = join(dir, 'landmarks.json')
  const output = join(dir, 'features.json')
  writeFileSync(input, JSON.stringify({ cases: cases.map(({ name, frames, length }) => ({ name, frames, length })) }))
  execFileSync(python(), [join(REPO, 'model', 'parity.py'), input, output], { cwd: join(REPO, 'model'), stdio: 'pipe' })
  const out = JSON.parse(readFileSync(output, 'utf8')) as { cases: { name: string; features: number[][] }[] }
  for (const c of out.cases) pyFeatures.set(c.name, c.features)
}, 120_000)

function maxDiff(ts: Float32Array, rows: number[][], L: number): number {
  expect(rows.length).toBe(L)
  let m = 0
  rows.forEach((row, i) => {
    expect(row.length).toBe(FEATURES_PER_FRAME)
    row.forEach((v, j) => {
      const a = ts[i * FEATURES_PER_FRAME + j]
      expect(Number.isFinite(a) && Number.isFinite(v)).toBe(true)
      m = Math.max(m, Math.abs(a - v))
    })
  })
  return m
}

describe('features.ts matches prepare.py', () => {
  it('found test cases at every length', () => {
    expect(cases.length).toBeGreaterThanOrEqual(11 * LENGTHS.length)
    expect(new Set(cases.map((c) => c.name)).size).toBe(cases.length)
  })

  for (const c of cases) {
    it(c.name, () => {
      const L = c.length ?? SEQUENCE_LENGTH
      const frames = c.frames.map(frameFromNested)
      const ts = computeFeatures(frames, L)
      expect(ts.length).toBe(L * FEATURES_PER_FRAME)
      const d = maxDiff(ts, pyFeatures.get(c.name)!, L)
      expect(d, `max |ts - py| = ${d}`).toBeLessThanOrEqual(TOL)
      if (c.expected) {
        const dk = maxDiff(ts, c.expected, L)
        expect(dk, `max |ts - kaggle| = ${dk}`).toBeLessThanOrEqual(TOL)
      }
      if (c.expectMirrored !== undefined) expect(stage1(frames).mirrored).toBe(c.expectMirrored)
    })
  }
})
