// Feature tensor -> top-k sign matches, with onnxruntime-web (wasm) and public/model/model.onnx.
// The model's input is [batch, L, F]: L (resample length) is read from the model itself, so a v3 model (L=32)
// and a v4 model (L=64) both work; pass `model.length` to computeFeatures. The model outputs probabilities
// (softmax is part of the exported graph). Shared with the extension.
import * as ort from 'onnxruntime-web/wasm'
// The .wasm binary as a Vite asset: served from node_modules in dev, emitted to dist/assets in the build
// (offline either way). Only the .wasm is given: this "bundle" build embeds its JS loader, and giving a
// wasmPaths *prefix* would make ORT import() a separate .mjs, which Vite refuses to serve from /public.
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { FEATURES_PER_FRAME } from './features'

ort.env.wasm.wasmPaths = { wasm: ortWasmUrl }
// One thread: multi-threading needs cross-origin isolation (COOP/COEP headers), and the model is small.
ort.env.wasm.numThreads = 1

const PAGE = 65536 // bytes per WebAssembly memory page
/** iPhone/iPad (iPadOS reports itself as a Mac with touch). */
export const isIOS = (): boolean =>
  typeof navigator !== 'undefined' &&
  (/iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1))
/** Largest memory ORT may reserve on iOS: 512 MB, far more than this 17 MB model needs. */
const IOS_MAX_PAGES = 8192
const MIN_PAGES = 2048 // 128 MB
/** What ORT's memory ended up as, for the debug panel. */
export const memoryInfo = { maximumMB: 0, retries: 0 }

/**
 * ORT's wasm build always creates a *shared* WebAssembly.Memory with a 4 GB maximum, even with one thread.
 * iOS Safari often refuses that ("RangeError: Out of memory": onnxruntime#22086, WebKit bug 255103), and the
 * model then never loads. While ORT starts up, shared memories get a smaller maximum: capped on iOS, and
 * halved on any browser that refuses the size asked for. Other memories (MediaPipe's) are untouched.
 */
async function withSafeSharedMemory<T>(start: () => Promise<T>): Promise<T> {
  const Native = WebAssembly.Memory
  const Safe = function (d: WebAssembly.MemoryDescriptor) {
    if (!d?.shared) return new Native(d)
    let max = d.maximum ?? 65536
    if (isIOS()) max = Math.min(max, IOS_MAX_PAGES)
    for (;;) {
      max = Math.max(max, d.initial)
      try {
        const m = new Native({ ...d, maximum: max })
        memoryInfo.maximumMB = (max * PAGE) / 2 ** 20
        return m
      } catch (e) {
        if (!(e instanceof RangeError) || max <= Math.max(MIN_PAGES, d.initial)) throw e
        memoryInfo.retries++
        max = Math.max(MIN_PAGES, d.initial, max >> 1)
      }
    }
  } as unknown as typeof WebAssembly.Memory
  Safe.prototype = Native.prototype
  WebAssembly.Memory = Safe
  try {
    return await start()
  } finally {
    WebAssembly.Memory = Native
  }
}

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

  /** `modelBytes`: model.onnx already downloaded (e.g. with progress); otherwise ORT fetches it from `base`. */
  static async load(base = `${import.meta.env.BASE_URL}model/`, modelBytes?: Uint8Array): Promise<SignModel> {
    const [session, labels] = await Promise.all([
      withSafeSharedMemory(() =>
        modelBytes
          ? ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'] })
          : ort.InferenceSession.create(`${base}model.onnx`, { executionProviders: ['wasm'] }),
      ),
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
    return this.topKAveraged([features], k)
  }

  /**
   * Test-time averaging: run several inputs for the same clip (e.g. trimmed variants) in ONE batched model call
   * and average their probabilities. With one input this is a plain lookup.
   */
  async topKAveraged(inputs: Float32Array[], k = 5): Promise<SignMatch[]> {
    const size = this.length * FEATURES_PER_FRAME
    const n = inputs.length
    if (n === 0) throw new Error('no inputs')
    const batch = new Float32Array(n * size)
    inputs.forEach((x, i) => {
      if (x.length !== size) throw new Error(`expected ${size} feature values, got ${x.length}`)
      batch.set(x, i * size)
    })
    const input = new ort.Tensor('float32', batch, [n, this.length, FEATURES_PER_FRAME])
    const out = await this.session.run({ [this.session.inputNames[0]]: input })
    const probs = out[this.session.outputNames[0]].data as Float32Array
    const C = probs.length / n
    const avg = new Float32Array(C)
    for (let i = 0; i < n; i++) for (let c = 0; c < C; c++) avg[c] += probs[i * C + c] / n
    return Array.from(avg.keys())
      .sort((a, b) => avg[b] - avg[a])
      .slice(0, k)
      .map((i) => ({ index: i, label: this.labels[i], gloss: this.labels[i].toUpperCase(), confidence: avg[i] }))
  }

  /** For tests / Node: wrap an existing session (load() is the browser path). */
  static fromSession(session: ort.InferenceSession, labels: string[]): SignModel {
    const meta = session.inputMetadata[0]
    const L = meta.isTensor ? meta.shape[1] : undefined
    if (typeof L !== 'number') throw new Error('model input has no fixed frame length')
    return new SignModel(session, labels, L)
  }
}
