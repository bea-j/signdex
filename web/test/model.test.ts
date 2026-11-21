// End-to-end check of the browser lookup path on real data: raw Kaggle landmarks -> features.ts at the frame
// length the model declares -> onnxruntime-web (wasm, as in the browser). The probabilities must match running
// the same model on the features Kaggle computed for those sequences (model/parity_fixture.json).
// Skips when model/model.onnx or the fixture is missing (no trained model yet).
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import * as ort from 'onnxruntime-web/wasm'
import { describe, expect, it } from 'vitest'
import { computeFeatures, FEATURES_PER_FRAME } from '../src/features'
import { frameFromNested, type NestedFrame } from '../src/layout'

const REPO = fileURLToPath(new URL('../..', import.meta.url))
const MODEL = join(REPO, 'model', 'model.onnx')
const FIXTURE = join(REPO, 'model', 'parity_fixture.json')
const have = existsSync(MODEL) && existsSync(FIXTURE)

describe.skipIf(!have)('model.onnx in onnxruntime-web on features.ts output', () => {
  it('reads L from the model and reproduces Kaggle-feature probabilities from raw landmarks', async () => {
    ort.env.wasm.numThreads = 1
    const session = await ort.InferenceSession.create(readFileSync(MODEL), { executionProviders: ['wasm'] })
    const meta = session.inputMetadata[0]
    const [, L, F] = meta.isTensor ? meta.shape : []
    expect(typeof L).toBe('number')
    expect(F).toBe(FEATURES_PER_FRAME)
    const labels = JSON.parse(readFileSync(join(REPO, 'model', 'labels.json'), 'utf8')) as string[]
    const fx = JSON.parse(readFileSync(FIXTURE, 'utf8')) as {
      cases: { sign: string; raw_xy: NestedFrame[]; features: number[][] }[]
    }
    const run = async (x: Float32Array) =>
      (await session.run({ features: new ort.Tensor('float32', x, [1, L as number, F as number]) })).probs
        .data as Float32Array
    for (const c of fx.cases) {
      const ours = computeFeatures(c.raw_xy.map(frameFromNested), L as number)
      const pOurs = await run(ours)
      if (c.features.length === L) {
        const pKaggle = await run(Float32Array.from(c.features.flat()))
        let d = 0
        for (let i = 0; i < pOurs.length; i++) d = Math.max(d, Math.abs(pOurs[i] - pKaggle[i]))
        expect(d, `max prob diff for "${c.sign}"`).toBeLessThan(1e-4)
      }
      let sum = 0
      for (const v of pOurs) sum += v
      expect(sum).toBeCloseTo(1, 4)
      const top = pOurs.indexOf(Math.max(...pOurs))
      console.log(`"${c.sign}" -> top-1 "${labels[top]}" ${pOurs[top].toFixed(2)} (L=${L})`)
    }
  }, 60_000)
})
