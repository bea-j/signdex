// Feature tensor -> top-k sign matches, with onnxruntime-web (wasm) and public/model/model.onnx.
// The model's input is [batch, L, F]: L (resample length) is read from the model itself, so a v3 model (L=32)
// and a v4 model (L=64) both work; pass `model.length` to computeFeatures. The model outputs probabilities
// (softmax is part of the exported graph). Shared with the extension.
import * as ort from 'onnxruntime-web/wasm'
import { FEATURES_PER_FRAME } from './features'

export interface SignMatch {
  index: number
  /** Kaggle label, e.g. "callonphone". */
  label: string
  /** Display gloss, e.g. "CALLONPHONE". */
  gloss: string
  confidence: number
}

/** Below this top-1 confidence the UI says it is not sure. */
export const LOW_CONFIDENCE = 0.4

export class SignModel {
  private constructor(
    private readonly session: ort.InferenceSession,
    readonly labels: string[],
    /** Frames per input (L), from the model's input shape. */
    readonly length: number,
  ) {}

  static async load(base = `${import.meta.env.BASE_URL}model/`): Promise<SignModel> {
    ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/` // served locally (scripts/copy-model.mjs)
    const [session, labels] = await Promise.all([
      ort.InferenceSession.create(`${base}model.onnx`, { executionProviders: ['wasm'] }),
      fetch(`${base}labels.json`).then((r) => {
        if (!r.ok) throw new Error(`labels.json: HTTP ${r.status}`)
        return r.json() as Promise<string[]>
      }),
    ])
    const dims = (i: ort.InferenceSession.ValueMetadata) => (i.isTensor ? i.shape : [])
    const [, L, F] = dims(session.inputMetadata[0])
    const [, C] = dims(session.outputMetadata[0])
    if (typeof L !== 'number' || L < 1) throw new Error(`model input has no fixed frame length: ${JSON.stringify(dims(session.inputMetadata[0]))}`)
    if (F !== FEATURES_PER_FRAME) throw new Error(`model expects ${F} features per frame, feature_spec.json gives ${FEATURES_PER_FRAME}`)
    if (typeof C === 'number' && C !== labels.length) throw new Error(`model has ${C} classes but labels.json has ${labels.length}`)
    return new SignModel(session, labels, L)
  }

  /** features: Float32Array of length * FEATURES_PER_FRAME from computeFeatures(frames, this.length). */
  async topK(features: Float32Array, k = 5): Promise<SignMatch[]> {
    const expected = this.length * FEATURES_PER_FRAME
    if (features.length !== expected) throw new Error(`expected ${expected} feature values, got ${features.length}`)
    const input = new ort.Tensor('float32', features, [1, this.length, FEATURES_PER_FRAME])
    const out = await this.session.run({ [this.session.inputNames[0]]: input })
    const probs = out[this.session.outputNames[0]].data as Float32Array
    return Array.from(probs.keys())
      .sort((a, b) => probs[b] - probs[a])
      .slice(0, k)
      .map((i) => ({ index: i, label: this.labels[i], gloss: this.labels[i].toUpperCase(), confidence: probs[i] }))
  }
}
