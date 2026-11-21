// lookupSign with test-time averaging: correctness against the single-pass path, plus timing for 1/3/5 variants
// on real Kaggle landmark clips with onnxruntime-web (wasm, 1 thread, as in the browser). Skips without a model.
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import * as ort from 'onnxruntime-web/wasm'
import { describe, expect, it } from 'vitest'
import { computeFeatures } from '../src/features'
import { frameFromNested, type NestedFrame } from '../src/layout'
import { lookupSign, ttaOffsets } from '../src/lookup'
import { SignModel } from '../src/model'

const REPO = fileURLToPath(new URL('../..', import.meta.url))
const MODEL = join(REPO, 'model', 'model.onnx')
const FIXTURE = join(REPO, 'model', 'parity_fixture.json')
const have = existsSync(MODEL) && existsSync(FIXTURE)

describe('ttaOffsets', () => {
  it('starts with the original clip and moves start/end inward by ~6%', () => {
    expect(ttaOffsets(40, 1)).toEqual([[0, 0]])
    expect(ttaOffsets(40, 3)).toEqual([[0, 0], [2, 0], [0, 2]])
    expect(ttaOffsets(40, 5)).toEqual([[0, 0], [2, 0], [0, 2], [4, 0], [0, 4]])
    expect(ttaOffsets(5, 5)).toEqual([[0, 0]]) // too short to trim
  })
})

describe.skipIf(!have)('lookupSign with test-time averaging', async () => {
  // model.ts points ORT at a Vite asset URL for the browser; in Node use ORT's own file next to its module.
  ort.env.wasm.wasmPaths = undefined
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(readFileSync(MODEL), { executionProviders: ['wasm'] })
  const labels = JSON.parse(readFileSync(join(REPO, 'model', 'labels.json'), 'utf8')) as string[]
  const model = SignModel.fromSession(session, labels)
  const clips = (JSON.parse(readFileSync(FIXTURE, 'utf8')) as { cases: { sign: string; raw_xy: NestedFrame[] }[] })
    .cases.map((c) => ({ sign: c.sign, frames: c.raw_xy.map(frameFromNested) }))

  it('with averaging off, matches the plain single-pass lookup exactly', async () => {
    for (const c of clips) {
      const plain = await model.topK(computeFeatures(c.frames, model.length), 5)
      const r = await lookupSign(model, c.frames, { variants: 1, k: 5 })
      expect(r.variants).toBe(1)
      expect(r.matches.map((m) => m.index)).toEqual(plain.map((m) => m.index))
      r.matches.forEach((m, i) => expect(m.confidence).toBeCloseTo(plain[i].confidence, 6))
    }
  })

  it('averages 3 and 5 variants into valid probabilities', async () => {
    for (const n of [3, 5]) {
      for (const c of clips) {
        const r = await lookupSign(model, c.frames, { variants: n, k: 250 })
        expect(r.variants).toBe(Math.min(n, ttaOffsets(r.stage1.T, n).length))
        const sum = r.matches.reduce((s, m) => s + m.confidence, 0)
        expect(sum).toBeCloseTo(1, 4)
      }
    }
  })

  it('timing: 1 vs 3 vs 5 variants (median of 40 lookups)', async () => {
    const median = (a: number[]) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)]
    for (let i = 0; i < 5; i++) await lookupSign(model, clips[0].frames, { variants: 5 }) // warm up
    const rows: string[] = []
    for (const n of [1, 3, 5]) {
      const total: number[] = []
      const feat: number[] = []
      const mod: number[] = []
      for (let i = 0; i < 40; i++) {
        const r = await lookupSign(model, clips[i % clips.length].frames, { variants: n })
        total.push(r.timing.totalMs)
        feat.push(r.timing.featuresMs)
        mod.push(r.timing.modelMs)
      }
      rows.push(`variants=${n}: total ${median(total).toFixed(1)} ms (features ${median(feat).toFixed(1)}, model ${median(mod).toFixed(1)})`)
    }
    for (const c of clips) {
      const one = (await lookupSign(model, c.frames, { variants: 1 })).matches[0]
      const five = (await lookupSign(model, c.frames, { variants: 5 })).matches[0]
      rows.push(`"${c.sign}": 1 pass -> ${one.label} ${one.confidence.toFixed(2)} | 5 averaged -> ${five.label} ${five.confidence.toFixed(2)}`)
    }
    console.log(`[lookup timing, L=${model.length}, onnxruntime-web wasm 1 thread]\n  ` + rows.join('\n  '))
  }, 120_000)
})
